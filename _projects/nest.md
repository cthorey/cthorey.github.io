---
layout: page
title: nest
description: What's happening at night on my brother front porch ?
img: /assets/img/nestimage.png
importance: 1
published: true
---

## Introduction 

Last month - I got a call from my brother. Long story short, a cat (and/or) dog was peeing on his front porch every night for the past few weeks; he needed some help to capture the evidence. In the following, I'll go through how to build a rig able to do just that using **ROS**, docker-compose and an old laptop webcam. 

The code is available [here](https://github.com/cthorey/nest).

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/signal-attachment-2020-05-10-230616.jpeg' | relative_url }}" alt="" title="example image"/>
    </div>
</div>


## Step 1 - Hardware

**First**, we need a camera than can see in the dark. We found an old laptop in his garage with a working webcam. Thankfully for us, a light turns on every time something is moving in his front porch so that should be enough :).


## Step 2 - Stream images from the camera

Now that we have a camera, we need to be able to stream images from it. To do that, we are going to use **[cv_camera](http://wiki.ros.org/cv_camera)**, a [ROS](https://www.ros.org/) package which will directly publish the images over the network for us (we could use opencv but where is the fun in that, right?). 

First, we need to install a udev rule for our camera and symlink it in $/sensors/camera$ to properly mount it in the container. Find the name of the camera using 
```
udevadm info /dev/video0
```
and fill in the name in the camera rule in **scripts/10-camera.rules**. Then, to install it, you can use **make install-udev-rules**. Great - now you should see your device under **/sensors/camera**. 

With this setup, we can just glue together the ros-master with the camera streamer using docker-compose

```
version: '2.0'

networks:
  ros:
    driver: bridge

services:
  ros-master:
    image: ros:kinetic-ros-core
    command: stdbuf -o L roscore
    networks:
      - ros
    restart: always

  camera:
    image: cthorey/nest
    devices:
      - /dev/sensors/uwrist_camera:/dev/sensors/camera
    depends_on:
      - ros-master
    environment:
      - "ROS_MASTER_URI=http://ros-master:11311"
      - "ROS_HOSTNAME=camera"
    command: ./scripts/start_camera.sh
    networks:
      - ros
```

## Step 3 - Detecting cats/dogs

Well, now that we have a streaming device, we need to figure out for each image if there is a cat and if so - grab his face and save it to disk. 

Given the images are published on the ROS network, we can just have a callback on each image
```
rospy.Subscriber("/cv_camera/image_raw", Image, rec.callback)
```
where callback is defined by

```
def callback(self, msg):
   img = pImage.fromarray(imgmsg_to_arr(msg))
   imgs = [img]
   preds = self.model.detect(imgs)
   if 'cat' in preds:
       rospy.loginfo('Detected a cat')
       self.save(img, 'cat')
   self.detection_idx += 1
   if self.detection_idx % 100 == 0:
       rospy.loginfo('We have been running {} detections so far'.format(
       self.detection_idx))
```

To find out if there is a cat/god in the image, we'll use some pre-trained model - after all, [all AI are good at recognising cat, are they not ?](https://www.wired.com/2012/06/google-x-neural-network/).

Here is the Model code:

```
class Model(object):
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cls_to_label = json.load(
            open(os.path.join(ROOT_DIR, 'scripts/cls_to_id.json')))

    def detect(self, imgs):
        imgs = torch.stack([self.transform(img) for img in imgs], 0)
        with torch.no_grad():
            preds = self.model(imgs)
            probs = nn.functional.softmax(preds[0], dim=0)
        idxs = np.arange(1000)
        labels = self.cls_to_label.values()
        mask = np.isin(idxs, self.cls_to_label.keys())
        p = probs[mask].sum().item()
        if p > 0.25:
            return 'cat'
        return ''
```

which essentially
1. download the pre-trained classifier - resnet50
1. take the image 
2. transforms it the way the network expects
3. do a forward pass through the model to get the logits
4. softmax the logits to get a probability distributions above all classes
5. sum up the probability of all class that could correspond to cat/dog

Those models are pre-trained on ImageNet and there are actually quiet a few categories that we have to considers - dalmatian, coach dog, carriage dog, Synset: Egyptian cat .. Full list can be found [here](http://www.image-net.org/search?q=cat) and [dogs](http://www.image-net.org/search?q=dog). 

We will grab and save the image if the probability is larger than 25% --> low threshold you'll say but we don't want to miss it and it's well worth reviewing a few false positive manually. 

Now, we can just add this new service to the docker-compose and we good to go

```
  recorder:
    image: cthorey/nest
    depends_on:
      - ros-master
      - camera
    environment:
      - "ROS_MASTER_URI=http://ros-master:11311"
      - "ROS_HOSTNAME=recorder"
    command: python -u scripts/recorder.py --frequency=0.2
    volumes:
      - ./model:/root/.cache/torch/checkpoints
      - ./data:/workdir/data
    networks:
      - ros
```
The full compose file can be found [here](https://github.com/cthorey/nest/blob/master/docker-compose.yaml)

Just need to run 
```
docker-compose up
```
and come back in the morning :smirk_cat:

# Final though - did it work ? 

First time !!!

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/catdetect.png' | relative_url }}" alt="" title="example image"/>
    </div>
</div>





