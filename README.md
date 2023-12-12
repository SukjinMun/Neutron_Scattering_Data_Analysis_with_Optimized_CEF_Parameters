# Optimized__CEF_Parameters_for_Neutron_Scattering_Data_Analysis
<p align="center">
  <img src="https://github.com/roysjmun0317/Optimized_CEF_Parameters_for_Neutron_Scattering_Data_Analysis/assets/78396618/391697bb-1e53-4cb7-8f6f-2e5ff3a92f68" alt="imageex" width="60%" />
</p>
The provided code within Jupyter Notebook conducts a thorough examination of neutron scattering data associated with a specific crystalline structure. This process involves the utilization of crystal field theory to delve into the intricacies of the material's behavior. Initially, the code initializes and converts crystal field parameters sourced from an external reference, ensuring necessary values for subsequent computations are established. Throughout its execution, it compares theoretical spectra derived from the reference data with experimental datasets, evaluating their consistency and identifying any disparities. To isolate the desired signal, our dataset undergoes background subtraction. Following this, the code employs a variety of functions to fit the theoretical spectra to the experimental data, extracting crucial information such as peak positions and widths. Utilizing optimization techniques such as scipy.optimize.shgo, it iteratively fine-tunes crystal field parameters, notably focusing on enhancing alignment between calculated and observed energy spectra, particularly emphasizing the refinement of B20. The culmination of this analysis results in informative visualizations showcasing optimized spectra, peak positions, chi-squared values, and the correlation between B-parameters and chi-squared values. This meticulous approach facilitates a comprehensive exploration of neutron scattering data within the realm of crystal field theory, without divulging specific details regarding the particular compound under investigation.

<pre>
*Note:
Please note that the raw neutron scattering data file used for this analysis is not included in this repo!
</pre>
