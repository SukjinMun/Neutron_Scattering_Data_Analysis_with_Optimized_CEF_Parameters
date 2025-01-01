# Neutron Scattering Data Analysis with Optimized CEF Parameters

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e64795e-53b0-430e-aed8-3ebb3954ce54" 
       alt="Neutron Scattering Data" 
       width="500" />
</p>
This Python script examines neutron scattering data for a specific crystalline structure by utilizing crystal electric field computations to explore complex material behaviors. It initializes and converts crystal field parameters from external references, and this ensures that the necessary values are correctly set for subsequent computations. The code then compares theoretical spectra derived from these references to experimental datasets and applies background subtraction to isolate meaningful signals and identify discrepancies. Next, it fits the theoretical spectra to the experimental data and captures peak positions and widths. A significant portion of the algorithm is dedicated to iteratively optimizing crystal field parameters to enhance the alignment between calculated and observed energy spectra. Finally, the script provides visualizations of the resulting spectra, chi-squared values, and the relationship between B-parameters and fit quality.
<pre>
*Note:
Original neutron scattering data file used for this analysis is not disclosed in this repo.
</pre>
