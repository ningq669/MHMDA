# AMHMDA

MHMDA is an innovative biologically interpretable multi-hop metapath and HeteroHyperNet (HHN) learning approach for predicting miRNA-disease associations. MHMDA follows three primary steps to achieve accurate miRNA-disease association identification. First, a "similarity-association-similarity" multi-hop meta-path learning method is introduced to capture specific associated pathway information by leveraging hierarchical attention perception, thereby connecting potentially associated miRNAs and diseases. Second, the HHN learning approach is employed to integrate both heterogeneous network and hyper network structures, effectively learning both direct and potential association information between miRNAs and diseases. Finally, the combination of the multi-hop meta-path with hierarchical attention and HHN allows for comprehensive learning of miRNA-disease associations from both local and global perspectives, significantly enhancing the richness and accuracy of the information. 






# Requirements
  * Python 3.7 or higher
  * PyTorch 1.8.0 or higher
  * torch-geometric 2.0.4
  * GPU (default)

# Data
  * Download associated data and similarity data.
  * Multiple similarity calculations are detailed in the supplementary material.

# Running  the Code
  * Execute ```python main.py``` to run the code.
  * Parameter state='valid'. Start the 5-fold cross validation training model.
  * Parameter state='test'. Start the independent testing.

# Note
```
 1.Torch-geometric has a strong dependency, so it is recommended to install a matching version.
 2.The trained model are stored in folder named cross valid . You can import directly to implement valid and test.
```
