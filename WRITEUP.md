# Project Write-Up

## Custom Layers in OpenVINO

### What are custom layers?
Custom layers are the layers that are not included into a list of known layers. This list of known layers is different for each of the supported frameworks. If your topology contains any layers that are not in the listof known layers, the Model Optimizer classifies them as custom.

### Process of converting custom layers
The process behind converting custom layers involves adding extensions to the Model Optimizer and the Inference Engine. For this,
1. The first step is to use the Model Extension Generator tool The MEG is going to create templates for Model Optimizer extractor extension, Model Optimizer operations extension, Inference Engine CPU extension and Inference Engine GPU extension.
2. Once customized the templates, next step is to generate the IR files with the Model Optimizer.
3. Finally, before using the custom layer in your model with the Inference Engine, you must: first, edit the CPU extension template files, second, compile the CPU extension and finally, execute the model with the custom layer.
### Reason for handling custom layers
In order to reduce the inference time OpenVINO Toolkit has its own list of optimized supported layers. The first step in generating an optimized Intermetidate Representation (IR) of the model is to convert all the layers of the input model into the supported layers. Even though the list contains a lot of different types of layer, it is not an exhaustive list. Researchers around the globe are contributing in creating new layers for their use cases. This leaves scope for future develpoment of OpenVINO Toolkit support development.
Some of the potential reasons for handling custom layers are:
1. As mentioned above, new layers are created as needed in research.
2. Different framework may have different implementation of a given layer that may result in framework dependend layer conversion. This means one layer in one framework may be supported and may not be supported in another framework. I would assume that this is a work in progress at Intel regarding layer support in multiple frameworks.
3. Custom layers also provide independence to explore other layers in OpenVINO framework.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were:
1. Based on [this](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) link, I wrote a script using TensorFlow framework to read a video file, perform predictions on it (detect people) and save individual frames with bounding boxes. This notebook is added in the project submission.
2. For calculating accuracy, I used the video provided in the `resources` folder, i.e., `Pedestrian_Detect_2_1_1.mp4`, which has 1394 frames.
3. Tensorflow script was run in Google colab environment and the required testing video was uploaded for testing. After completion output folder with all frames was zipped and downloaded for inspection on local machine.
4. I then performed testing using manual inspection marking any misclassified frame as 0 and all other as 1 in an excel sheet. This sheet is also added in the project submission.
**Note:** A frame with **zero people** and **zero bouding box** in it, is marked as 1, which is the reason for a high accuracy.
5. Following tables summarize the findings for three models explored
- **without the use of the OpenVINO™ Toolkit**

| Model Name               | ssd_mobilenet_v1 | ssd_mobilenet_v2 | ssd_inception_v2 |
|--------------------------|------------------|------------------|------------------|
| Load Time (s)            | 1.321108         | 1.824082         | 1.832367         |
| Inference Time (s)/frame | 0.098251         | 0.13675          | 0.235019         |
| Size (MB)                | 29.1             | 69.7             | 102              |
| Accuracy (%)             | 86.37            | 95.91            | 92.04            |
- **with the use of the OpenVINO™ Toolkit**

| Model Name               | ssd_mobilenet_v1 | ssd_mobilenet_v2 | ssd_inception_v2 |
|--------------------------|------------------|------------------|------------------|
| Load Time (s)            | 0.236901         | 0.449007         | 0.727141         |
| Inference Time (s)/frame | 0.045075         | 0.070126         | 0.157916         |
| Size (MB)                | 27.3             | 67.4             | 100.2            |
| Accuracy (%)             | 87.3             | 95.27            | 88.67            |

6. Pre-conversion accuracy and times are based on model execution on google colab, whereas, post-conversion accuracy and times are based on model execution on Udacity workspace

## Assess Model Use Cases

Some of the potential use cases of the people counter app are counting number of people entering any concert, museum, shopping mall, etc.

Each of these use cases would be useful because this would let the user know how many people have entered or left the building/location.
- Let's say we may want to reduce the people capacity to 50% (for example) for a locaiton, an app like this would play a significant role in desiging such system for crowd control.
- Let's say we install this system on an isle in a warehouse, it can provide useful information about how long does people usually spend time figuring out waht to buy. The possibilities are endless, where there is a need to count people this app will provide a reasonable baseline.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:
- As we will see in the model research section model accuracy will affect the accuracy of the app in terms of people counted as well as in the estimation of the average duration. Different models behaved differently.
- Lighting will affect the image captured by the camera, this may in turn affect the model accuracy.
- Camera focal length and image size may affect the image resolution and it is possible that the model may or may not detect a person. This will in turn affect the model accuracy.
- Since these models are trained on COCO dataset, if we get an image that is outside the range of quality of trained image, we will get unpredictable results. Model may or may not perform accordingly.
- Any other parameter that can affect the image captured will also affect the model accuracy, such as, occlusion, glare, rain, etc.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: **ssd_mobilenet_v1_coco**
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)

  ```
  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    ```
    ```
    tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    ```
  - I converted the model to an Intermediate Representation with the following arguments

  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py
  --input_model ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
  --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v1_coco_2018_01_28/pipeline.config
  --reverse_input_channels
  --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  --output_dir ssd_mobilenet_v1_coco_2018_01_28
  ```
  - The model was insufficient for the app because it was calculating average time spend per person as 0.01 sec and total number of people counted as 41 with reduced threshold.
  - I tried to improve the model for the app by decreasing the threshold value to 0.2. To make sure that app predicts only one bounding box I plotted the bounding box with maximum score, in case there were more than one predicted bounding boxes.
  - Another way to improve the app could be to have a buffer counter and keep the value of count alive if there are missing boxes for few frames. This trick might have worked if the consecutive number of missing frames are significantly less than the number of frames with no people in the video. For example, the minimum number of frames in between transition with no person in frame are 35 and the maxmimum number of consecutive misclassified frames are ~50 with this model.


- Model 2: **ssd_mobilenet_v2_coco**
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  ```
  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  ```

  ```
  tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  ```
  - I converted the model to an Intermediate Representation with the following arguments
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py
  --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
  --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
  --reverse_input_channels
  --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  --output_dir ssd_mobilenet_v2_coco_2018_03_29
  ```


  - The model was insufficient for the app because it was calculating average time spend per person as 0.04 sec and total number of people counted as 27 with reduced threshold.
  - I tried to improve the model for the app by decreasing the threshold value to 0.2. To make sure that app predicts only one bounding box I plotted the bounding box with maximum score, in case there were more than one predicted bounding boxes.
  - Interestingly the minimum number of frames in between transition with no person in frame are 35 and the maxmimum number of consecutive misclassified frames are ~10 with this model. If we inlude a buffer of 10 consecutive frame in the code, this model would be a suitable candidate for the app.
  - I choose to explore another model with even larger size, assuming that might resolve the bounding box flickering issue.

- Model 3: **ssd_inception_v2_coco**

  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)

  ```
  wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
  ```

  ```
  tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz
  ```
  - I converted the model to an Intermediate Representation with the following arguments
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py
  --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb
  --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config
  --reverse_input_channels
  --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  --output_dir ssd_inception_v2_coco_2018_01_28
  ```
  - The model was insufficient for the app because it was calculating average time spend per person as 0.06 sec and total number of people counted as 28 with reduced threshold.
  - Contrary to my assumptions, this model was taking more time to perform inference in real time and accuracy was also compromised w.r.t. the second model.
- Based on the above study, model 2 was the most promissing candidate for the people counter app.
- However, I choose to go ahead with an existing Intermediate Representations provided by Intel®.
- Model 4: **person-detection-retail-0013**
  - This model can be accessed using the model downloader. The model downloader downloads the .XML and .BIN files that will be used by the application.
  ```
  sudo ./downloader.py --name person-detection-retail-0013
  ```
- Here comes the kicker, I was hoping that this model would not have flicker but the flicker issue is still persistant. Now, I went back to solving the flicker problem.
- I added a fixed length queue to track detections, if there is any detection present in the queue, its sum will be > 0 and this condition will serve as if condition to my program to reassign the people counted to be one (I am assuming there is just one person/frame in the video).
- Adding a queue solved the problem, now I have good results with both `person-detection-retail-0013` and `ssd_mobilenet_v2_coco` models.

## Run the application

  Assuming all the dependencies are installed and the toolkit is correctly sourced, you can use the following steps to run the application, see [link](https://github.com/intel-iot-devkit/people-counter-python) for more information:

### Step 1 - Start the Mosca server

  ```
  cd webservice/server/node-server
  node ./server.js
  ```
  You should see the following message, if successful:
  ```
  Mosca server started.
  ```
### Step 2 - Start the GUI
  Open new terminal and run below commands.

  ```
  cd webservice/ui
  npm run dev
  ```
You should see the following message in the terminal.
```
webpack: Compiled successfully

```
###  Step 3 - FFmpeg Server
  Open new terminal and run the below commands.
```

sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Run the code
  Open a new terminal to run the code.

**Setup the environment**

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```
You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.

**Running on the CPU**

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at: `/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/`

Depending on whether you are using Linux or Mac, the filename will be either libcpu_extension_sse4.so or libcpu_extension.dylib, respectively. (The Linux filename may be different if you are using a AVX architecture)

  Though by default application runs on CPU, this can also be explicitly specified by -d CPU command-line argument:

  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```
