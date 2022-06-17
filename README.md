# ncovid-autoregressive-forecast
Source-code for the autoregressive containers of ncovid framework

## Install

Clone the repository:

    git clone https://github.com/Natalnet/ncovid-autoregressive-forecast.git

Run:

    cd ncovid-autoregressive-forecast/src
    conda env create -f environment.yml
    conda activate ncovid-ar-forecast

This environment uses miniconda3.

## Testing the API with Postman

- Select the Postman agent as 'Desktop agent'
- Run the first cell in on-line_predicting.ipynb 
- Send requests with Postman. Use the *train_ncovid_* and *predict_ncovid_* requests [here](https://go.postman.co/workspace/Team-Workspace~c466ad9c-c9b9-41da-87fe-a445382bc6be/collection/16914400-1f99ace5-4924-43ff-b28d-0edc4fca8892?action=share&creator=16914400).
