# 🛒 E-Commerce Product Recommendation System

## Overview

This project is an E-Commerce Product Recommendation System developed using Machine Learning and Streamlit. The system recommends products to users using two different recommendation techniques:

* Collaborative Filtering (SVD)
* Content-Based Filtering (TF-IDF + Cosine Similarity)

The application provides personalized product recommendations, product similarity suggestions, and evaluation metrics through an interactive dashboard.

---

## Features

### 📊 Dashboard Analytics

* Total Users
* Total Products
* Total Ratings
* Ratings Distribution Visualization

### 🤝 Collaborative Filtering

* Uses Singular Value Decomposition (SVD)
* Predicts ratings for products not yet rated by a user
* Generates Top 5 personalized product recommendations

### 🏷️ Content-Based Filtering

* Uses product metadata:

  * Category
  * Brand
  * Price Range
  * Tags
* Applies TF-IDF Vectorization
* Computes similarity using Cosine Similarity
* Recommends similar products based on user preferences

### 📈 Model Evaluation

The recommendation system is evaluated using:

* Precision@5
* Recall@5
* NDCG@5

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Scikit-Surprise

---

## Project Workflow

### Step 1: Data Loading

The application loads:

* ratings.csv
* products.csv

### Step 2: Dashboard Creation

Displays:

* Number of Users
* Number of Products
* Number of Ratings
* Ratings Distribution Chart

### Step 3: Collaborative Filtering

* Train SVD recommendation model
* Predict ratings for unseen products
* Recommend top products to users

### Step 4: Content-Based Filtering

* Combine product attributes
* Generate TF-IDF vectors
* Calculate product similarity matrix
* Recommend similar products

### Step 5: Evaluation

Evaluate recommendation quality using:

* Precision@5
* Recall@5
* NDCG@5

---

## Dataset Structure

### ratings.csv

| Column     | Description            |
| ---------- | ---------------------- |
| User_ID    | Unique user identifier |
| Product_ID | Product identifier     |
| Rating     | User rating (1–5)      |

### products.csv

| Column      | Description           |
| ----------- | --------------------- |
| Product_ID  | Product identifier    |
| Category    | Product category      |
| Brand       | Product brand         |
| Price_Range | Product price segment |
| Tags        | Product keywords      |

---

## Project Structure

```text
ecommerce-product-recommendation-system/
│
├── app.py
├── products.csv
├── ratings.csv
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shahnewajmazumder/ecommerce-product-recommendation-system.git
```

Move into the project directory:

```bash
cd ecommerce-product-recommendation-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

After running the command, open the local URL displayed in the terminal.

---

## Application Functionalities

### Collaborative Filtering Recommendations

1. Enter a User ID.
2. Click "Get Collaborative Recommendations".
3. View the top recommended products and predicted ratings.

### Content-Based Recommendations

1. Select a product previously rated by the user.
2. Click "Get Content-Based Recommendations".
3. View the most similar products with similarity scores.

---

## Evaluation Metrics

### Precision@5

Measures how many recommended products are relevant.

### Recall@5

Measures how many relevant products are successfully recommended.

### NDCG@5

Measures the ranking quality of recommendations.

---

## Future Improvements

* Hybrid Recommendation System
* Real-Time User Interaction Tracking
* Product Popularity Recommendations
* Deep Learning-Based Recommender Models
* Cloud Deployment using Streamlit Cloud

---

## Author

**Shah Newaj Ahshan Mazumder**

Data Analytics Intern

---

## License

This project is created for educational and internship purposes.
