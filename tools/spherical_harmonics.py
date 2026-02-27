import numpy as np
import math
from envmap import EnvironmentMap
from pyshtools.backends import shtools
import matplotlib.pyplot as plt


class SphericalHarmonic:
    def __init__(self, coeffs, norm=4):
        """
        Initializes the SphericalHarmonic object directly from coefficients.

        It's recommended to use the class methods `from_envmap` or `from_array`
        to create instances of this class.

        :param coeffs: A numpy array of shape (3, N_sh) where N_sh is the number
                       of spherical harmonic coefficients.
        :param norm: The normalization of the coefficients. `4` means orthonormal.
        """
        if not isinstance(coeffs, np.ndarray) or coeffs.ndim != 2 or coeffs.shape[0] != 3:
            raise ValueError("coeffs must be a numpy array of shape (3, N_sh).")

        self.coeffs = coeffs
        self.norm = norm
        self.n_coeffs = self.coeffs.shape[1]

        # Automatically detect the order (lmax) from the number of coefficients.
        # N_sh = (lmax + 1)^2
        lmax_plus_1 = int(np.sqrt(self.n_coeffs))
        if lmax_plus_1**2 != self.n_coeffs:
            raise ValueError(
                f"The number of coefficients ({self.n_coeffs}) is not a perfect square, "
                "so the SH order cannot be determined."
            )
        self.lmax = lmax_plus_1 - 1

    @classmethod
    def from_envmap(cls, envmap_input, copy_=True, max_l=None, norm=4):
        """
        Projects an environment map to its spherical harmonics basis.

        :param envmap_input: An EnvironmentMap object or a numpy array for the map.
        :param copy_: If True, copies the input data.
        :param max_l: The maximum degree of the expansion.
        :param norm: The normalization of the coefficients. `4` means orthonormal.
        """
        spatial = envmap_input.copy() if copy_ else envmap_input

        if not isinstance(spatial, EnvironmentMap):
            spatial = EnvironmentMap(spatial, 'LatLong')

        if spatial.format_ != "latlong":
            spatial = spatial.convertTo("latlong")

        coeffs_list = []
        for i in range(spatial.data.shape[2]):  # Iterate over R, G, B channels
            cilm = shtools.SHExpandDH(spatial.data[:, :, i], norm=norm, sampling=2, lmax_calc=max_l)
            vec = shtools.SHCilmToVector(cilm)
            coeffs_list.append(vec)

        # Return a new class instance
        return cls(np.asarray(coeffs_list), norm=norm)

    @classmethod
    def from_array(cls, coeffs_array, norm=4):
        """
        Creates a SphericalHarmonic object from a numpy array of coefficients.

        :param coeffs_array: A numpy array of shape (3, N_sh).
        :param norm: The normalization of the coefficients. `4` means orthonormal.
        """
        # Return a new class instance
        return cls(coeffs_array, norm=norm)

    def reconstruct(self, height=None, max_l=None, clamp_negative=True, apply_window=False):
        """
        Reconstructs the spatial environment map from the SH coefficients.

        :param height: The height of the reconstructed lat-long map.
        :param max_l: The maximum degree to use for reconstruction. If None,
                      uses all available coefficients.
        :param clamp_negative: If True, clamps reconstructed values to be non-negative.
        """
        lmax_recon = self.lmax if max_l is None else min(max_l, self.lmax)

        # To get an output grid of height H, we must pass lmax = (H/2) - 1.
        # This requires the height to be an even number.
        if height % 2 != 0:
            raise ValueError(f"Grid height ({height}) must be an even number for DH sampling.")

        # This is the correct lmax to generate a grid of the desired size.
        grid_lmax = (height // 2) - 1

        retval = []
        for i in range(self.coeffs.shape[0]):  # Iterate over R, G, B channels
            vec = self.coeffs[i, :]
            cilm = shtools.SHVectorToCilm(vec, lmax=self.lmax)
            if apply_window:
                cilm = self.window(cilm)
            grid = shtools.MakeGridDH(cilm, norm=self.norm, sampling=2, lmax=grid_lmax, lmax_calc=lmax_recon)
            retval.append(grid)

        retval = np.asarray(retval, dtype=np.float32).transpose((1, 2, 0))

        if clamp_negative:
            retval = np.maximum(retval, 0)

        return retval

    def window(self, cilm, function="sinc"):
        """
        Applies a windowing function to the coefficients to reduce ringing artifacts.
        See https://www.ppsloan.org/publication/StupidSH36.pdf

        :param cilm: The coefficient array of shape (2, lmax+1, lmax+1).
        :param function: The windowing function to use.
        """
        current_lmax = cilm.shape[1] - 1
        if current_lmax == 0:  # Avoid division by zero for L=0
            return cilm

        if function == "sinc":
            # Create a kernel k(l/L) for each degree l
            x = np.arange(current_lmax + 1) / current_lmax
            kernel = np.sinc(x)
        else:
            raise NotImplementedError(f"Windowing function {function} is not implemented.")

        # Apply the kernel value for each degree l to all orders m
        for l in range(current_lmax + 1):
            cilm[:, l, :] *= kernel[l]

        return cilm

    def visualize_dominant_light_direction(self, output_path, height, max_l=None, tonemap=False):
        """
        Visualizes the dominant light direction from the order 1 SH coefficients
        on the reconstructed environment map using the correct, verified coordinate system.

        :param output_path: Path to save the output image.
        :param height: The height of the reconstructed lat-long map.
        :param max_l: The maximum degree to use for reconstruction.
        """
        if self.lmax < 1:
            print("Cannot visualize dominant light direction, lmax is less than 1.")
            return

        # Reconstruct the environment map
        envmap = self.reconstruct(height=height, max_l=max_l)
        if tonemap:
            envmap = envmap ** (1/2.2)  # Simple gamma tonemapping

        # Use luminance coefficients for brightness to handle colored light correctly
        c_r, c_g, c_b = self.coeffs
        c_lum = 0.2126 * c_r + 0.7152 * c_g + 0.0722 * c_b
        c1, c2, c3 = c_lum[1], c_lum[2], c_lum[3]

        #
        # --- The Correct, Verified Coordinate System Mapping ---
        # Discovered through systematic visualization and analysis.
        # This mapping correctly aligns the pyshtools SH coefficients with the
        # 3D Cartesian coordinate system of the environment map.
        x = -c2
        y = -c3
        z = c1
        #
        # --- End of Mapping ---
        #

        # Normalize the direction vector
        direction = np.array([x, y, z])
        norm = np.linalg.norm(direction)
        if norm == 0:
            print("Dominant light direction has zero magnitude.")
            # Optionally save the map without a cursor
            plt.imsave(output_path, np.clip(envmap, 0, 1))
            return
        direction /= norm

        # Convert Cartesian coordinates to spherical coordinates (in radians)
        theta = np.arccos(direction[2])  # Colatitude
        phi = np.arctan2(direction[1], direction[0])  # Longitude

        # Convert to degrees
        lat = 90.0 - np.rad2deg(theta)
        lon = np.rad2deg(phi)

        # Map longitude and latitude to pixel coordinates
        width = 2 * height
        u = (lon + 180.0) / 360.0 * width
        v = (90.0 - lat) / 180.0 * height

        # Plot the reconstructed environment map and the cursor
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        envmap_clip = np.clip(envmap, 0, 1)
        ax.imshow(envmap_clip)
        ax.plot(u, v, 'r+', markersize=50, markeredgewidth=8)  # A large red cross
        ax.axis('off')  # Turn off the axes
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # The following two methods are for debugging and verifying the coordinate system.
    def visualize_axis_permutations(self, output_path, height, max_l=None):
        """
        Visualizes all 6 possible permutations of mapping SH coefficients c1, c2, c3
        to the Cartesian axes x, y, z. This is used to definitively find the correct
        coordinate system for the SH library being used.

        :param output_path: Path to save the output image.
        :param height: The height of the reconstructed lat-long map.
        :param max_l: The maximum degree to use for reconstruction.
        """
        if self.lmax < 1:
            print("Cannot visualize, lmax is less than 1.")
            return

        # Reconstruct the environment map
        envmap = self.reconstruct(height=height, max_l=max_l)

        # Use luminance coefficients for brightness
        c_r, c_g, c_b = self.coeffs
        c_lum = 0.2126 * c_r + 0.7152 * c_g + 0.0722 * c_b
        c1, c2, c3 = c_lum[1], c_lum[2], c_lum[3]

        # --- Define the 6 possible axis permutations ---
        # The key is the mapping "xyz", the value is the tuple of coefficients (c1, c2, c3)
        # that are assigned to (x, y, z) respectively.
        permutations = {
            "x=c3, y=c1, z=c2": (c3, c1, c2),  # The common graphics convention we assumed
            "x=c1, y=c2, z=c3": (c1, c2, c3),
            "x=c1, y=c3, z=c2": (c1, c3, c2),
            "x=c2, y=c1, z=c3": (c2, c1, c3),
            "x=c2, y=c3, z=c1": (c2, c3, c1),
            "x=c3, y=c2, z=c1": (c3, c2, c1),
        }

        # Setup for plotting
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(envmap)
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(permutations)))
        width = 2 * height

        # --- Loop through, calculate, and plot each permutation ---
        # For this test, we will use positive signs for all components.
        # This will get us to the correct quadrant.
        for i, (label, (x, y, z)) in enumerate(permutations.items()):
            direction = np.array([x, y, z])
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm

            # Convert Cartesian to Spherical to Pixel coordinates
            theta = np.arccos(direction[2])
            phi = np.arctan2(direction[1], direction[0])
            lat = 90.0 - np.rad2deg(theta)
            lon = np.rad2deg(phi)
            u = (lon + 180.0) / 360.0 * width
            v = (90.0 - lat) / 180.0 * height

            # Plot a colored marker with its label
            ax.plot(u, v, 'P', markersize=15, markeredgewidth=2.5, color=colors[i], label=label)

        # Finalize the plot
        ax.set_title('Axis Permutation Candidates for Dominant Light Direction')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(title="Mappings (x, y, z) <-> (c1, c2, c3)", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Make room for legend
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved axis permutation visualization to {output_path}")

    def visualize_final_candidates(self, output_path, height, max_l=None):
        """
        Based on the deduction that z=c1, this function tests the remaining
        permutations and sign combinations for the x and y axes to find the
        definitive coordinate system mapping.

        :param output_path: Path to save the output image.
        :param height: The height of the reconstructed lat-long map.
        :param max_l: The maximum degree to use for reconstruction.
        """
        if self.lmax < 1:
            print("Cannot visualize, lmax is less than 1.")
            return
        
        max_l = min(max_l, self.lmax) if max_l is not None else self.lmax

        # Reconstruct the environment map
        envmap = self.reconstruct(height=height, max_l=max_l)

        # Use luminance coefficients for brightness
        c_r, c_g, c_b = self.coeffs
        c_lum = 0.2126 * c_r + 0.7152 * c_g + 0.0722 * c_b
        c1, c2, c3 = c_lum[1], c_lum[2], c_lum[3]

        # --- Define the 8 final candidates based on your deduction z=c1 ---
        candidates = {
            # Case 1: x=c2, y=c3
            "x= +c2, y= +c3, z=c1": (+c2, +c3, c1),
            "x= -c2, y= +c3, z=c1": (-c2, +c3, c1),
            "x= +c2, y= -c3, z=c1": (+c2, -c3, c1),
            "x= -c2, y= -c3, z=c1": (-c2, -c3, c1),
            # Case 2: x=c3, y=c2
            "x= +c3, y= +c2, z=c1": (+c3, +c2, c1),
            "x= -c3, y= +c2, z=c1": (-c3, +c2, c1),
            "x= +c3, y= -c2, z=c1": (+c3, -c2, c1),
            "x= -c3, y= -c2, z=c1": (-c3, -c2, c1),
        }

        # Setup for plotting
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(envmap)
        # Use distinct colors and markers
        colors = plt.cm.jet(np.linspace(0, 1, 8))
        markers = ['P', 'P', 'P', 'P', 'X', 'X', 'X', 'X']

        width = 2 * height

        # --- Loop through, calculate, and plot each candidate ---
        for i, (label, (x, y, z)) in enumerate(candidates.items()):
            direction = np.array([x, y, z])
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm

            # Convert Cartesian to Spherical to Pixel coordinates
            theta = np.arccos(direction[2])
            phi = np.arctan2(direction[1], direction[0])
            lat = 90.0 - np.rad2deg(theta)
            lon = np.rad2deg(phi)
            u = (lon + 180.0) / 360.0 * width
            v = (90.0 - lat) / 180.0 * height

            # Plot a colored marker with its label
            ax.plot(u, v, marker=markers[i], markersize=12, markeredgewidth=2.5, color=colors[i], label=label)

        # Finalize the plot
        ax.set_title('Final Candidates for Dominant Light Direction (Assuming z=c1)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(title="Mappings", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # Make room for legend
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved final candidate visualization to {output_path}")