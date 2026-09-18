def format_eval_video_name(
    task_str,
    variation,
    demo_id,
    reward,
    source_episode_number=None,
):
    episode_suffix = (
        f"_episode{source_episode_number}"
        if source_episode_number is not None
        else ""
    )
    return (
        f"{task_str}_var{variation}_demo{demo_id}"
        f"{episode_suffix}_reward{reward}.mp4"
    )
