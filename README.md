# Neutron Scattering Data Analysis with Optimized CEF Parameters

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e64795e-53b0-430e-aed8-3ebb3954ce54" 
       alt="Neutron Scattering Data" 
       width="500" />
</p>


This Jupyter Notebook demonstrates analyzing neutron scattering data for a specific crystalline structure, utilizing crystal field theory to examine complex material behaviors. It begins by initializing and converting crystal field parameters from external references, ensuring that necessary values are correctly set for subsequent calculations. It then compares theoretical spectra derived from these references to experimental datasets, applying background subtraction to isolate meaningful signals and identify discrepancies. Next, the notebook fits the theoretical spectra to the experimental data, capturing insights such as peak positions and widths. A key component is iteratively optimizing crystal field parameters (with a focus on B20) via scipy.optimize.shgo to enhance alignment between calculated and observed energy spectra. Finally, the demo provides visualizations of refined spectra, chi-squared values, and the relationship between B-parameters and fit quality. This delivers a comprehensive overview while preserving confidentiality about the molecule under investigation. Note that this example is only a partial demonstration of the full code.
<pre>
*Note:
Original neutron scattering data file used for this analysis is not disclosed in this repo.
</pre>
