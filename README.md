# Introduction to Waas-MR Integrated Assessment Meta-Model
Welcome to the repository of the Waas-MR Integrated Assessment Meta-Model. 
The model has been used to test the Dynamic Adaptive Policy Pathways framework for multi-risk systems (DAPP-MR)

## :bulb: Core Features of the Waas-MR model
* Sector-Specific Impact Analysis: We focus on three crucial sectors: agriculture, urban development, and shipping. Waas-MR simulates the impacts from flood and drought events on these sectors as well as the effects of a wide range of Disaster Risk Management measures which sectors implement depending on how the future unfolds.
* Accounting for interdependencies: This model explicitly captures interaction effects between floods and droughts, as well as between the sectors and their respective DRM strategies. The model can be run in different modes of complexity (traditional siloed, perspective, multi-hazard risk and multi-risk)
* High-Resolution Simulation: The Waas-MR model operates at a spatial resolution of 100m x 100m. It uses 10-day timesteps to simulate changes and impacts and accumulate data over a 100-year planning horizon.

[Schematisation of the Waas MR model with different sectoral impact modules to
stress-test pathways](Misc/WaasMR.eps)

## Regarding the publication
**Julius Schlumberger<sup>1,2,*</sup>, Marjolijn Haasnoot<sup>1</sup>, Jeroen C.J.H. Aerts<sup>1,2</sup>, Veerle Bril<sup>2</sup>, Lars van der Weide<sup>2</sup>, Marleen de Ruiter<sup>2</sup>**

<sup>1</sup> Deltares, Boussinesqweg 1, 2629 HV Delft, the Netherlands </br>
<sup>2</sup> Institute for Environmental Studies (IVM), Vrije Universiteit Amsterdam, De Boelelaan 1111, 1081 HV Amsterdam, the Netherlands

<sup>*</sup> Corresponding author: [julius.schlumberger@deltares.nl](julius.schlumberger@deltares.nl)

To cite work related to this repository, please use the following citation

```
Schlumberger, J., Haasnoot, M., Aerts, J., & de Ruiter, M. (2022). Proposing DAPP-MR as a disaster risk management pathways framework for complex, dynamic multi-risk. iScience, 25 (10), 105219. doi: 10.1016/j.isci.2022.105219
```


# Replicate our experiments and recreate our figures
This GitHub repository contains the code and data needed to replicate the computational experiments and recreate all figures for Schlumberger et al. (in review). 
Note that this experiment was run using high performance computing. The version shared allows to run the model on a personal computer.

_Note: Only the post-processed results are provided due to memory constraints. Please contact the corresponding author at julius.schlumberger@deltares.nl for full set of model outputs._

## :package: Create an Environment with all required Packages**
The WaasMR model has been developed in Python 3.9. To create the appropriate environment, please refer to [environment.yml](environment.yml).</br>
After downloading this repository and enabling the correct intepreter, you should be able to run the model from `run_model.py`. All filepaths are relative to the main project folder (WaasMR).  

## :file_folder: Folder Structure
* `waasmodel_v6` contains all scripts and inputs required to run the model. It includes files to specify the inputs, module-specific classes and functions, the Waasmodel architecture and a wrapper to be able to run and store the outputs.
* `PotentialPathways` contains input files used for the stress-testing. They contain all options of sector-specific pathway (combination) that will be stresstested. E.g. 1,0,3,4 specifies that flood-agriculture pathway 1 in combination with no pathway for drought-agriculture, in combination with flood-urban pathway 3 and in combination with drought-shipping pathway 4 will be used as input for the stress-testing run.
* `postprocessing` contains additional scripts to check the completeness of the stress-testing run, along with scripts to generate endvalues and robustness indicators for the evaluation criteria
* `viz` contains scripts required for the creation of the figures in Schlumberger et al. (_in review_).
* `model_outputs` is the location where model outputs and figures will be stored.

## :building_construction: Possible Modifications
The available code can be used to replicate the results presented and discussed in Schlumberger et al. (_in review_). Without much effort, the available code can be used to explore the model and its results. 
`run_model.py` allows to run the model for a specific pathways combination, for different stages of the analysis and different sector-hazard perspectives. It can also be used to run the model by implementing an initial measure at the start of the planning horizon.
If additional pathways should be tested, refer to the input files `waasmodel_v6/inputs`.

_Note: Refer to the [Supplemental Information](Misc/SupplementalInformation.pdf) for a high-level conceputalization of the model structure._

# Contribution Guidelines
We warmly welcome contributions from the community and are excited to see how you can help improve the Waas-MR Integrated Assessment Meta-Model! Whether you're fixing bugs, adding new features, or improving documentation, your help is invaluable. Here's how you can contribute:
1. Bug Reports and Feature Requests: If you find a bug or have an idea for a new feature, please check the Issues section first to see if it has already been reported or suggested. If not, feel free to open a new issue, providing as much detail as possible.
2. Submitting Pull Requests (PRs): Create a new branch in your fork for your changes. Keep your changes as focused as possible. If you are addressing multiple issues, it’s better to create separate branches and PRs for each.

If you have any questions or need help with setting up, feel free to reach out to us (provide contact details or link to a discussion forum).

# Licensing Information
This model and all information - if not stated differently - are available via CC BY 4.0. If you are interested in further developing or using the WaasMR model, please feel free to reach out to the corresponding author to explore options for collaboration.

# Acknowledgements
JS, MH, and MdR have been supported in this research by the European Union’s Horizon 2020 research and innovation programme (grant no. 101003276) as part of the MYRIAD\_EU project. JA has been supported in this work by the ERC grant COASTMOVE, grant 884442. The work reflects only the author’s view and the agency is not responsible for any use that may be made of the information it contains. We thank SURF (www.surf.nl) for the support in using the National Supercomputer Snellius.\\

The authors wish to acknowledge Kukuh Wachyu Bias, coloripop, Adrien Coquet, Hary Murdiono JS, Amelia, Luis Prado, Alex Burte, shashank singh, Eucalyp, Symbolon, Ian Rahmadi Kurniawan, Ralf Schmitzer, Andre Buand, ProSymbols, M. Oki Orlando, and Nurul Hotimah for the icons made available through (https://thenounproject.com/) used in our figures, which are available under the Creative Commons Attribution 3.0 Unported License (CC BY 3.0). 

