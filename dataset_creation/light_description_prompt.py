# --- Prompt Definitions ---
indoor_prompt_dict = {
    "one_paragraph":
    """
    Start directly, begin the paragraph by immediately describing the most significant light source. Do not use introductory sentences like "The scene is illuminated by..." or "There are several light sources."
    1. Identify Key Light Sources and Describe Each Source: Use the direct light source position and brightness information provided below, and use the panorama for full context, identify the dominant light sources that directly illuminate the scene in the cropped image. These can include windows, lamps, strip lights, fixtures, or other things to your knowledge. For each significant and dominant light source, describe its type, position relative to the view (similar to which in the direct light information), color, and brightness. Only one concise sentence should be used for each light source.
    2. Use one short and concise sentence to describe the overall color of the scene.
    """,
}

outdoor_prompt_dict = {
    "one_paragraph":
    """
    Start directly, begin the paragraph by immediately describing the most significant light source. Do not use introductory sentences like "The scene is illuminated by..." or "There are several light sources."
    1. Describe the Primary Natural Light: Identify the main source of natural light with the light information provided below. In a single sentence, describe its direction relative to the view (similar to which in the direct light information), its color/hue, and its brightness.
    2. Detail Any Artificial Lights: If any artificial lights are active and visible (like streetlights or building lights), briefly describe their type, location, and color.
    3. Use one short and concise sentence to describe the overall color of the scene.
    """
}


def get_light_description_prompt(scene_type, detail_level='low_one_paragraph'):

    if detail_level == 'low_one_paragraph':
        instruction = "Based on the provided images (cropped view, panorama, and coordinate map), write a concise, single-paragraph description of the lighting as seen from the perspective of the cropped image. \n"
        prompt_dict = indoor_prompt_dict if scene_type == 'indoor' else outdoor_prompt_dict
        user_prompt_template = f"""{instruction}
        The following is the content of this paragraph:
        {prompt_dict['one_paragraph']}
        
        Important formatting requirements:
        - Must give the correct and faithful description based on the lighting conditions of the scene
        - Do not mention the coordinate colors in your final output.
        - Make sure this paragraph is straightforward and flows naturally, and avoid redundancy
        - Write in complete sentences without using bullet points, dashes, or numbered lists
        - Do not use bullet points, dashes (-), or numbered lists
        - Provide concise and brief descriptions
        - Do not use words expressing uncertainty like 'appears to be', 'seems to', 'likely', or 'suggests'. State the lighting conditions as fact
        - Do not use words like 'cropped image', 'cropped view', 'panorama' in your final output
        
        You analysis should:
        Use the Panorama for Context: the panorama provides a complete 360-degree view of all light sources. Use this to understand the lighting, but focus your description only on the lights that directly and strongly illuminate the scene in the cropped image.
        Use the direct light source position and brightness information below (very important, the most precise information) to understand the lighting conditions
        """
    else:
        raise ValueError(f"Unsupported detail level: {detail_level}")

    return user_prompt_template
