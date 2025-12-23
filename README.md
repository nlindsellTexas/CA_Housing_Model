# California Housing Price Prediction

This project predicts California housing prices using the California Housing dataset. 
We explore feature engineering, train a Ridge regression model, and evaluate its performance.

## Dataset

- Source: California Housing dataset
- Contains 20,640 rows and 9 features:
  - longitude, latitude
  - housing_median_age
  - total_rooms, total_bedrooms
  - population, households
  - median_income
  - ocean_proximity
- Target: `median_house_value` (capped at $500,000)

## Features

- Added engineered features:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:YourUsername/CA_Housing_Model.git
   cd CA_Housing_Model