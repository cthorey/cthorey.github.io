---
layout: page
title: driven-data competition
description: Clog Loss Advance Alzheimer's Research with Stall Catchers competition
img: /assets/img/clogloss0.png
importance: 1
---

# Competiton summary

This competition was organized by [driven data](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/207/) and aims to predict if a blood vessels is clogged or not from a video clip of the vessel itself. 

<blockquote>
5.8 million Americans live with Alzheimer’s dementia, including 10% of all seniors 65 and older. 
Scientists at Cornell have discovered links between “stalls,” or clogged blood vessels in the brain, and Alzheimer’s. Stalls can reduce overall blood flow in the brain by 30%. The ability to prevent or remove stalls may transform how Alzheimer’s disease is treated.
</blockquote>

## benchmark

The data comes in the form of an heavily unbalanced set of video clip (99.7 flowing / 0.3 stalled).

As a quick and dirty benchmark - I first tried to sample $$N$$ frame for each video clip, extract
the part underline in orange and build a 10x10 mosaic using this snippet 

```
def batch_to_image(batch):
    n = len(batch)
    w, h = batch[0].shape
    arr = np.vstack((batch, np.zeros((100 - n, w, h))))
    arr = [arr[j * 10:j * 10 + 10] for j in range(10)]
    final = []
    for a in arr:
        final.append(a.ravel().reshape((w * 10, w)).T)
    return Image.fromarray(np.vstack(final).astype('uint8'))


def process_one(filename, size=64):
    vid = imageio.get_reader(filename, 'ffmpeg')
    imgs = []
    for i in range(vid.count_frames()):
        frame = vid.get_data(i)
        mask = cv2.inRange(frame, (255, 69, 0), (255, 180, 0))
        mask = np.argwhere(mask)
        xmin, xmax, ymin, ymax = mask[:, 0].min(), mask[:, 0].max(
        ), mask[:, 1].min(), mask[:, 1].max()
        img = Image.fromarray(
            frame[xmin:xmax, ymin:ymax, :]).convert('L').resize((size, size))
        imgs.append(np.array(img))
    return batch_to_image(imgs)
```

Here is a sample of three clogged vessels

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/Screenshot from 2020-10-24 16-56-48.png' | relative_url }}" alt="" title="example image"/>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/Screenshot from 2020-10-24 16-56-54.png' | relative_url }}" alt="" title="example image"/>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/Screenshot from 2020-10-24 16-56-48.png' | relative_url }}" alt="" title="example image"/>
    </div>
</div>

Using the resulting mosaic - I decided to just finetune a head on top of a resnet18 backbone. Using a weighted cross-entropy loss to take care of the unbalanced dataset, I manage to get an MCC~0.2 - not too bad but still far off the top of the leaderboard.

## Introducing slowfast

[SlowFast](https://arxiv.org/pdf/1812.03982v3.pdf) if an architecture developped by FAIR for video recognition. At the time of the competition, it was the leading model on [here](https://paperswithcode.com/task/video-recognition).

The core idea is to break the processing in two different pathways:
1. One **fast** pathway ingesting a high FPS but lightweight to capture temporal information
2. A **slow** pathway ingesting only a few frame focusing on semantic. 

They release the entire codebase such that, after some tiny modification on my [end](https://github.com/cthorey/SlowFast), I was able to train the network almost as is on the dataset. 

Using this architecture, the performance were closer to 50% MCC. 

## Conclusion

I trained only one model using SlowFast and this is the model that was scored against the private leaderboard. At the end of the day, it got 50% MCC - enough to be in the top percent of the participant 21/922 participants but still far off the winning solution which achieves 85%!


