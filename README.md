# Neutron Scattering Data Analysis with Optimized CEF Parameters

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e64795e-53b0-430e-aed8-3ebb3954ce54" 
       alt="Neutron Scattering Data" 
       width="500" />
</p>
This Python script offers a detailed examination of neutron scattering data for a specific crystalline structure, leveraging crystal field theory to explore complex material behaviors. The workflow begins by initializing and converting crystal field parameters from external references, ensuring that the necessary values are correctly set for subsequent computations. The code then compares theoretical spectra derived from these references to experimental datasets, applying background subtraction to isolate meaningful signals and identify discrepancies. Next, it fits the theoretical spectra to the experimental data, capturing key insights such as peak positions and widths. A significant portion of the algorithm is dedicated to iteratively optimizing crystal field parameters (particularly focusing on B20), using scipy.optimize.shgo to enhance the alignment between calculated and observed energy spectra. Finally, the script provides visualizations of the resulting spectra, chi-squared values, and the relationship between B-parameters and fit quality, delivering a comprehensive analysis while preserving confidentiality regarding the molecule under investigation.
<pre>
*Note:
Original neutron scattering data file used for this analysis is not disclosed in this repo.
</pre>
