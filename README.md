# Medical-image-reconstruction-and-synthesis
This code is developed for medical image reconstruction and synthesis based on proposed framework.
We include two tasks in the file, i.e., low-dose PET reconstruction and PET-CT synthesis.
Small dataset are provided to demo the code.

System requiements:
We need system intalled Pytorch and ODL package.

Training:
In each tasak, there are three stage as mentioned in the paper. User needs to train each model seperatively. We also provide our well-trained models.


Testing:
Users need to copy the best model to the '.\Test\Saved_Model', and run test.py to inference. 
We also provide the well-trained model in the folder.
User can download our provided data and model. By filling the right path of data, user can get test resutls in '.\Test\Saved_Model' by simply run 'test.py'.
