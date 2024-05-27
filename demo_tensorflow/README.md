# Learning Fused Pixel and Feature-based View Reconstructions for Light Fields (FPFR)
<br>
Welcom to our FPFR project! This project includes a simplified version of our program demonstrating how our approach generates a light field based on the sparsely sampled views. We strongly recommand that the readers follow the instructions below to synthesize a light field by yourself. The code source will be available soon.

## Check environment
Our project is implemented in **Python** with the deep learning framework **tensorflow (version 1.13.1)**

Before launching the demo code, please make sure that the following packages are correctly installed.   
Check list: **tensorflow**,**numpy**,**matplotlib**

## Launch demo
We offered several light field scenes to test our demo code, running the `run_demo.sh` script to launch our demo. The command in `run_demo.sh` is:  
`python demo.py --scene_name=stilllife --mode=FPFR* --data_type=synthetic --angular_resolution=7 --inter_extra=inter`  
- ***scene_name*** is the name of light field scene.  
- ***mode*** offers two mode, FPFR refers to a simple prediction, and FPFR* refers to the average of several predictions (more details in our paper).  
- ***data_type*** indicates the type of light field data (synthetic or lytro).  
- ***inter_extra*** indicates a demo for view interpolation or extrapolation (inter or extra).  
- ***angular_resolution*** is the angular resolution of the generated light field, in the *inter* mode, angular resolution should be an integer greater or equal to 3; in the *extra* mode, angular resolution should be an odd integer greater than 3.  

## Other informations
We offered several test scenes in the folder **scenes**, if you successfully launched our program, bravo! The synthesized light field can be found in the folder **results**.  
Thank you for your attention, we sincerely hope you enjoy our project.  
