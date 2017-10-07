# Feature-Extraction-using-AlexNet
A practice of feature extraction using AlexNet.

## Sample Outputs

### ImageNet Inference
- Run command to see AlexNet (pretrained on ImageNet) out on two sample images of Poodle and Weasle.
    
    ```
    python imagenet_inference.py

    ```
- You should see output like this

	```
	Image 0
	miniature poodle: 0.389
	toy poodle: 0.223
	Bedlington terrier: 0.173
	standard poodle: 0.150
	komondor: 0.026
	
	Image 1
	weasel: 0.331
	polecat, fitch, foulmart, foumart, Mustela putorius: 0.280
	black-footed ferret, ferret, Mustela nigripes: 0.210
	mink: 0.081
	Arctic fox, white fox, Alopex lagopus: 0.027
	
	Time: 0.117 seconds
	```

### Traffic Sign Inference on AlexNet pretrained on ImageNet
- Run command to see AlexNet (pretrained on ImageNet) out on two sample images of traffic sign _stop_ and traffic sign _construction_. 
    
    ```
    python traffic_sign_inference.py

    ```
- You should see output like this. **Note** your output will not math this exactly as your initial weights (probably random) will be different.

	```
	Image 0
	screen, CRT screen: 0.051
	digital clock: 0.041
	laptop, laptop computer: 0.030
	balance beam, beam: 0.027
	parallel bars, bars: 0.023
	
	Image 1
	digital watch: 0.395
	digital clock: 0.275
	bottlecap: 0.115
	stopwatch, stop watch: 0.104
	combination lock: 0.086
	
	Time: 0.127 seconds
	```
	
### Traffic Sign Inference with AlexNet feature extraction
- Run command to see inference with AlexNet feature extraction on two sample images of traffic sign _stop_ and traffic sign _construction_.
    
    ```
    python feature_extraction.py

    ```
- You should see output like this. **Note** your output will not math this exactly as your initial weights (probably random) will be different.

	```
	Image 0
	Dangerous curve to the left: 1.000
	End of no passing: 0.000
	End of speed limit (80km/h): 0.000
	Right-of-way at the next intersection: 0.000
	Speed limit (60km/h): 0.000
	
	Image 1
	Go straight or left: 1.000
	End of no passing: 0.000
	Turn right ahead: 0.000
	Priority road: 0.000
	Traffic signals: 0.000
	
	Time: 0.093 seconds
	```
	
### Train AlexNet with Feature Extraction on Traffic Sign Data
Here the last fully connected classification layer of AlexNet is replaced with classification layer for traffic signs. Run following command to start training.

```
python train_feature_extraction.py
```

Training AlexNet (even just the final layer!) can take a little while, so if you don't have a GPU, running on a subset of the data is a good alternative. As a point of reference one epoch over the training set takes roughly 53-55 seconds with a GTX 970.

