todos:

- [] Find related work:
  1. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5729374
  2. 

mention the usage of ssd in testing and things for ngtpy

## Examples:

1. Image lag for **ALGORITHM**:  

   ![image](/home/dumbmachine/code/SVMWSN/paper_related/things/FrameComparison.png)

   

# DeepQuery: Content Based Video Search & Retrieval

### Abstract:

Digital media, comprised largely of images and videos, has been on the rise in the past decade. Due to its ease of access and efficiency in spreading information, Pixels now have become the currency used by most information transaction on the Internet. 

We present **DeepQuery** a object detection based deep learning model which leverages the scenes from video clips to generate fixed sized vectors for videos. **These fixed sized vectors are used to retrieve videos from clips**. It offers a solution to efficient solution which balances the precision/accuracy with processing computation.

On both, UFC101 and **a custom dataset containing clips from various tv shows and movies** , generates competitive performance in terms of runtime and accuracy. We also show that benefits of using our `algorithm` in a constraint   (videos). Code is available at: https://github.com//dumbmachine/deepquery

Key words: Content-based video retrieval, Video Database, object detection, approximate nearest neighbour search.

### 1. Introduction:

Video search is a very challenging problem which not only requires alot of processing but storage as well. Internet has seen the sites like Google Images and Tineye which provide the facility to reverse image search, such facilities for reverse searching videos using small clips or even images are yet to see the sunshine. 

Large amount of information is stored in video files as multiple image frames. `Youtube gets over 500 hours of video uploaded every minute  [and maybe some other facts]`.  Various industry based archives also exist. 

Well-established methods for searching and retrieving video content rely on 

1. manually crafted features (Knowledge-Discounted Event Detection in Sports Video)
2. close captioning (DeepSeek: Content Based Image Search & Retrieval), 
3. audio annotations or using hash bits generated using visual features `give example for each`. Among these for Video hashes have been in research extensively due to their fast searching capabilities.

However, most current works in scalable visual search focus on image-based retrieval [6]–[14] or text-image/image-text retrieval [15]–[19]. To our best knowledge, there are only few works that present an efficient framework for video hashing.

`Video hash ki extra information`

Recently, deep CNN has been successfully applied on many vision tasks, such as image classification (Krizhevsky, Sutskever, and Hinton 2012) and image segmentation (Sharma, Tuzel, and Jacobs 2015), which shows the powerful ability of deep CNN for describing complex non-linear mappings and learning rich mid-level representations. The CNN features learned from data are discriminative, but still high dimensional for the retrieval task. . In retrieval tasks, the hashing-based methods which project high dimensional features to a low dimensional space are often used (Liu et al. 2012; Li et al. 2015b), but feature extraction and hash function learning in these methods are independent, in other words, the extracted features may not be compatible with the hash coding process. Some models use an end-to-end system for mixing the feature generation and its appropriate hash generation (Dong, Jia, Wu and Pei 2016). 

We present a novel solution for Content Based Video Retrieval, wherein the Objects, their respective positions and histogram are indexed. An object detection model is used to obtain inference of each frame. The inference information is usedcd to cluster the individual frames to obtain key frame representations of each scene of video. We define a "scene" as set of frames in video wherein the total change in information,  prominently the objects, their location and background , is minimal. This helps use to remove redundant frames while making sure nearly all unique frame information is not lost. 

Our proposed solution is highly scalable, the inference of video frames can easily be run on multiple gpus and the algorithm used for clustering of video frames is also highly parallelized. This system makes use of the resources available to it efficiently. Our searching algorithm also offer the advantage of `tables of indexes` reducing the number of comparisons by only comparing with frames having similar number of objects. ANN from ngtpy is used for comparison.

### 2. Related Work:

 In this section, we briefly review realted topics: 1) feature comparison based searching 2) video hashing.+

categorize the papers and give example and reference for each one of them`

### 3. Proposed Approach

In this section, we first present the basic idea and pipeline of this algorithm. Then the different elements of this cbvr system, object detector, k means-based frame object clustering and finally a fast searching method, are explained with further details.

1. Pipeline:

   The pipeline begins with a video. A object detector is used to get the information of objects from each frame in the video. The required information for each object are:

   The video is divided into segments or shorter clips, wherein each clip would give a certain threshold number of frames. The frames would be the key frames from that segment of the video. The key frame(s) from each segment would then be combined to obtain the shorter video representation of the original clip. It was observed for most videos the best segmentation strategy is to divide the video is buckets of bucket size as the fps of the video. 

   `Give maths here.`

   The object information from frames is used to determine the appropriate cluster for each frame in the image. The algorithm used for clustering is defined in details `mention section number here`. Videos `of one kind, ask ihab` have utmost 4 scene changes, here by scene changes we refer to change in the overall representation of frame where objects and their background are of utmost importance. Thus 4 cluster are chosen for clustering the frames of each segment. All the segment return 4 frames each. This shorter representation of the video is then indexed into a database. The objects and their respective location from each frame along with its background histogram are used in indexing. 

   Video frames may have different number of object in them, this results in each frame's vector representation having a different dimension, with a successive difference of 6. This serves as a very important part of the search algorithm as it isolates the vector with appropriate dimensions in-turn  reducing the total search computations. For further information about the searching refer to section `section number `.

A. Object Detection / Feature Extraction

This section describes the importance of choosing the correct Object detector for the pipeline.  

The object detector plays a very important role in the pipeline and thus the correct choice of object detector is important?. Accuracy and inference time are two major factors when considering object detectors. If the detector is unable to detect the objects in the image, this leads to loss in information for the vector representation. But highly accurate object detector suffers from very high inference times leading to increase in the time required to build data for indexing and also increases the search time. 

Thus SSD: Single Shot MultiBix Detector [here](https://arxiv.org/pdf/1512.02325.pdf) was chosen as the detector. Most  object detectors have approaches similar to:  hypothize the bounding boxes, resample pixels or features for each box, and apply a high-quality classifier. `reference for R-CNN.` or approach similar to: dividing the image into regions and predicts bounding boxes and probabilities for each region.  `refrence to yolo`.While accurate, these approaches have been too computational intensive for embedded systems and, even with high-end hardware, too slow for real-time applications. The impact of inference time has a drastic impact on the usability of this system. 

Thus SSD: Single Shot MultiBox Detector was chosen as the detector of choice. `reference SSD`. The fundamental improvement in speed comes from eliminating bounding box proposals and the subsequent pixels or features resampling stage. Usage of small convolutional filter to predict object categories and offsets in bounding boxes proposals, using separate predictors for different aspect ratio detectors, and applying these filters to multiple feature maps from the later stages of a network in order to perform detection at multiple scales.

Using an object detector with focus on inference time reduces the total number of indexable frames since the ability for a frame to be indexable depends on whether the object from the frame where detected or not. 

`fGive example of images where side shots of the object are not tagged human or something` 

To overcome such cases, we augmented the training dataset with additional data of common object, from daily life, from different angles. This helped increase the indexable `from this number to this number.`



`difference between coco and specially trained model, accuracy and total indexable frames`

`Give speed differences, seraching and indexing dataset difference for each of the objects detectors (Fast-RCNN, SSD, YoloV3)`

B. Frame Clustering

We use a K-Means algorithm for clustering the video frames into unique clusters. The features used for clustering are the 

`table of that thing`

confidence score, bounding boxes and category of each object in the video frame. The frames in one segment of video are clustered on the basis of the respective objects. Among the frames f~i~ in segment s~i~ , the highest number of objects are noted. Let this number be denoted by O~max~ and the video frame by f~max~ . Frames in S~i~ can have utmost O~max~  in any order. All the features of objects O~j~ , where j E {0..len(O~max~)}, is the j^th^ object in frame F~i~ , are then clustered using K-Means algorithm into `4` clusters. The number of clusters can be changed to increase the accuracy by trying to store as many distinct frames from the segment as possible, though increasing the number of frames per segment has drastic effects on the retrieval times. 

`table displaying the following.`

[format for algorithm](https://arxiv.org/pdf/1601.07754v1.pdf)

---------------------------------------------------
-------------------------------------------------------------------------------------------------------------

Algorithm 1 The proposed key frames extraction

--------------------------------------------------------------------------------------------------------

Input: V~i~
Output: 

FPS** = Fps of the video

**S** = {s1, s2, ... sN}: Segments from the video divided into intervals of its fps
**F** = {f~ij~}: Video frames, where i is the index of video segment and j (0<j<**FPS**) is the index of the video frame in the segment s~i~.
O = {o~ijk~}: Objects, where k is the index of the object in video frame~i~,in the segment~i~  

1. **for** i=1 to N step 1:
    F~i~ = GetFramesFromSegment(i)
    **for** j=1 to M step 1:
        f~ij~ = GetFrameFromFrames(j)
    
    ​    



`Explain each step from this`

### 4. Experiments

To evaluate the effectiveness of our proposed system for scalable content based video retrieval, we conducted experiments on 3 video datasets. Namely the UCF101, HDF51 and a custom `Rick and Morty dataset`. The details of the experiments and the results are descrived in the following sections.

A. Datasets and Experiments Settings



B. Pretraining:

: The normal ImageNet Pretraining and the second specific pretraining

[ref](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7797446&tag=1)

UCF101 dataset [ref](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/): It consists of 13320 videos with the average `number` seconds and `number` frames. It has  101 action categories such as Apply Eye Makeup, Apply Lipstick, Blow Dry Hair, Brushing Teeth and Cutting in Kitchen. We tested the precision of our system with different number of frames representing the whole video. Since most of the categories in this dataset are events, the videos contain large variation among frames making the task very challenging. In  our experiments, we selected all the indexable videos from the dataset, 30 video per category as the query data. 

For our object detector, we used 4 different models; SSD with COCO, SSD with MyCOCO, RCNN , `some SOTA`. We tested with 4 different frame representations sizes for segments of f = 2, 4, 6, all frames.

HMDB dataset [ref]: It consists of 6849 clips distributed into 51 action classes like chew, climb, hug. `difference and reason for low accuracy, objects from coco not seen properly `. In  our experiments, we selected all the indexable videos from the dataset, 02 video per category as the query data. 

B. Evaluation Metrics

To measure the performance of our `algorithm`, we Laplacian Distance ranking as evaluation metrics. For ranking, the mean Average Precision (mAP) and Precision@N are evaluation. 

C. Experimental Results

Comparison with different deep baselines: We first 