-------------------------------------------------------------
Stain Normalisation Toolbox for Matlab
Nicholas Trahearn and Adnan Khan
BIALab, Department of Computer Science, University of Warwick
-------------------------------------------------------------


-------------------------------------------------------------
0. Contents
-------------------------------------------------------------

    1. Version History
    2. System Requirements
    3. Usage Instructions
    4. Toolbox Contents
    5. Disclaimer
    6. Contact Information


-------------------------------------------------------------
1. Version History
-------------------------------------------------------------

    2.2 --- Released 28th May 2015
        - Added install.m, a script that automatically configures the toolbox for use.
        - Removed precompiled mex files of Landini's Colour Deconvolution code, due to cross-platform compatibility issues.
        - Minor bug fixes.

    2.1 --- Released 28th April 2015
        - Added the v1.0 executable version of the Non-Linear method as a separate function. 
              - This version can be called using the Matlab function NormSCDLeeds.m.
              - NormSCD.m will continue to call the v2.0 Matlab implementation.
              - NormSCDLeeds.m is only compatible with Windows Platforms.
              - NormSCDLeeds.m uses a built-in classifier and cannot be retrained.
        - Added Norm.m, a single function from which any normalisation method can be called. Also allows a custom normalisation method to be called from within the toolbox.
        - Minor bug fixes to the training functions for Non-Linear (Khan) Stain Normalisation.

    2.0 --- Released 26th February 2015
        - Replaced executable for Non-Linear (Khan) Stain Normalisation with a MATLAB implementation. 
        - Added functions to train custom classifiers for the Non-Linear (Khan) method (please refer to SCDTraining/README.txt for more details).
        - Removed redundant functions (AddThirdStainVector.m, Lab2RGB.m, and RGB2Lab.m) and replaced any references to them in code with the equivalent MATLAB built-in functions.
        - Separated Deconvolve.m into two functions: 
                Deconvolve.m            - Serves the same purpose as previously.
                PseudoColourStains.m    - Converts grayscale stain channels into pseudo-colour images, with respect to a given matrix. 
                                          All visualisation from Deconvolve.m is now made by a call to PseudoColourStains.m.
        - Renamed all Stain Normalisation functions and files to follow a consistent format (Norm______.m, where _____ is the name of the normalisation method).
        - Renamed the Macenko stain matrix estimation function and file to follow a consistent format for stain matrix estimation (EstUsing_______.m, where ______ is the stain matrix estimation method).

    1.0 --- Released 4th September 2014
        - Original release, containing MATLAB functions for RGB Histogram Specification, Reinhard, and Macenko methods of Stain Normalisation. 
        - Also includes an executable for the Non-Linear (Khan) Stain Normalisation method.


-------------------------------------------------------------
2. System Requirements
-------------------------------------------------------------

    In order to ensure that all features of the Toolbox function as intended we recommend the following:
        - Matlab version 2014a or later.
        - Image Processing Toolbox and Statistics Toolbox installed.
        - C compiler configured for use in Matlab.


-------------------------------------------------------------
3. Usage Instructions
-------------------------------------------------------------

    (1) Unzip the stain_normalisation_toolbox folder into your Matlab working directory.

    (2) Within Matlab, change your directory to the stain_normalisation_toolbox folder.

    (3) Run install.m to set up the toolbox.

    (4) Run demo.m to see a demonstration of the normalisation methods included in the toolbox.


-------------------------------------------------------------
4. Toolbox Contents
-------------------------------------------------------------

    The Stain Normalisation Toolbox currently contains the following components:

        (1) A MATLAB implementation of the Non-Linear Stain Normalisation algorithm reported in the following publication:

                Khan, A.M., Rajpoot, N., Treanor, D., Magee, D., A Non-Linear Mapping Approach to Stain Normalisation in Digital Histopathology Images using Image-Specific Colour Deconvolution, IEEE Transactions on Biomedical Engineering, 2014. 

            This method can be run from the MATLAB function NormSCD, please view SCDTraining/README.txt for more information about the supervised component of this method.

        (1a) An executable version of the Non-Linear Stain Normalisation algorithm, which can be run from the Matlab function NormSCDLeeds. This version is only available on Windows platforms.

        (2) MATLAB implementations of three more stain normalisation algorithms used in histological image analysis:

            a. RGB Histogram Specification (MATLAB function NormRGBHist).
            b. Reinhard Colour Normalisation (MATLAB function NormReinhard).
            c. Macenko Stain Normalisation (MATLAB function NormMacenko).

        (3) Matlab compatable C code for G. Landini's implementation of Ruifrok & Johnston's Colour Deconvolution algorithm. 


-------------------------------------------------------------
5. Disclaimer
-------------------------------------------------------------

    The toolbox and its components are provided 'as is' with no implied fitness for purpose. 
    The author is exempted from any liability relating to the use of this toolbox.  
    The toolbox and its components are provided for research use only.
    The toolbox and its components are explicitly not licenced for re-distribution (except via the websites of Warwick University and Leeds University).


-------------------------------------------------------------
6. Contact Information
-------------------------------------------------------------

    Please send all comments and feedback to Nicholas Trahearn at: N.Trahearn@warwick.ac.uk

