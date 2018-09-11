# AutoCat: Automated Product Categorization

## Overview
The goal of this project is to create a product classifier that will be used in product management software.  

Categorizing products on a product-by-product basis for thousands of products can be time consuming and error prone and is problem for which machine learning can be brought to bear.  Labelled data from existing businesses that belong to a data pool provides an opportunity to develop a machine learning model that is capable of automatically categorizing a set of existing, unlabelled products and labelling new products as they arrive in real time through electronic invoices.

By eliminating the need for new and existing users to manually categorize thousands of products, the onboarding process for product management software can be significantly improved; leading to a better user experience and improved sales and marketing efforts.

## Project Structure
    .
    ├── autocat             # Project source code
    │   ├── data            
    │   └── models          
    ├── data                # Raw and transformed data
    │   ├── interim
    │   ├── processed
    │   └── raw
    ├── models              # Saved models
    └── notebooks           # Notebooks
    └── Makefile            # Project tasks

## Project State
The current version of this project includes notebooks and application code to build machine learning models for the product classification task.  The project does not currently include the contents of the `data` folder and is missing the initial ETL and exploratory notebooks.  The missing folder and notebooks will be added once the source data has been sanitized.  In the meantime feel free to reach out if you have any questions about this codebase - calvindlm@gmail.com.