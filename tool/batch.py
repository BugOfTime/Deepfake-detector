
def parse_batch(batch):
    video = audio = labels = None
    if isinstance(batch, dict):
        video  = batch.get("video",  batch.get("frames"))
        audio  = batch.get("audio",  batch.get("sound"))
        labels = batch.get("label",  batch.get("labels"))
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            video, audio, labels = batch
        elif len(batch) == 2:
            data, labels = batch
            if isinstance(data, dict):
                video = data.get("video", data.get("frames"))
                audio = data.get("audio", data.get("sound"))
            else:
                video = data
        else:
            return None, None, None
    else:
        return None, None, None

    return video, audio, labels
