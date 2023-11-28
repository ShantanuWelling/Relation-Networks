## CS337: Artificial Intelligence and Machine Learning, Fall 2023, IIT Bombay

Link to presentation: <br/>
https://iitbacin-my.sharepoint.com/:p:/g/personal/210010076_iitb_ac_in/EVQ-hjYA8-RMvDoVA0aKA_cBuUYZbe0J_HOJMjPu7M4vfw?e=cBpTse <br/>
Also added at "AIML Presentation.pdf"

There are 5 folders in this repo. <br/>
cnn_mlp folder is the SOTA model (for comparison) on Sort-of-CLEVR data <br/>
rn_sort folder is Relation Network for Sort-of-CLEVR data <br/>
vgg folder is our improvement obtained on Sort-of-CLEVR data by changing the object generator CNN of the Relation Network to pretrained VGG. <br/>

To run the above 3 folders, navigate to the respective folder and run: <br/>
```python3 sort_of_clevr_generator.py``` <br/> 
and then run: <br/>
```python3 main.py```

This saves the model runs to folder named runs and model in the folder named model and results to a csv file inside the same directory. <br/>

The next 2 folders are: <br/>
RelationNetworks-CLEVR - Relation Network for CLEVR data (State Description input variant) <br/>
rn_clevr_pix - Relation Network for CLEVR data (Image input variant) <br/>

To run the above 2 folders, first download the CLEVR data from https://cs.stanford.edu/people/jcjohns/clevr/ , extract it and place it in the parent directory (i.e. the directory which contains the above 2 directories) <br/>
navigate to the respective folder and run: <br/>
```python3 main.py``` <br/>

This saves the outputs in folders "model*", "logs", "test_results" and the file logfile.log for RelationNetworks-CLEVR and in the folders- model, runs and a .csv file for rn_clevr_pix. 
