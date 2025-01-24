# Artifact-free fat-water separation in Dixon MRI using deep learning

To use, clone the repository, install the environment and run the following commands. 

EDIT: The model can be found here https://www.dropbox.com/scl/fi/17hjdyq2fsf2u74di9msg/fat_water_separation_model.pth?rlkey=wadss2nnxlytwki676wgqsk0m&st=1xl7nmwt&dl=0 (apologies there were some storage issues and the model file stopped working) 
```bash
cd fat_water_separation
python perform_fat_water_separation.py fat_water_separation_model.pth ip_path op_path
```
It will save the estimated fat and water channels to nifti files that can be used for analyses.

If you use this in your research please cite the below:

- Basty N, Thanaj M, Cule M, Sorokin EP, Liu Y, Thomas EL, Bell JD, Whitcher B. Artifact-free fat-water separation in Dixon MRI using deep learning. _Journal of Big Data_ **10**, 4 (2023). [DOI: 10.1186/s40537-022-00677-1][j_big_data]

[j_big_data]: https://doi.org/10.1186/s40537-022-00677-1
