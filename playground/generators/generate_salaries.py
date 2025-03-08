import pandas as pd
import numpy as np


def generate_salary_csv(filename: str, num_rows: int = 1000) -> None:
    """
    Generate a CSV file with sample salary data.

    The CSV will contain three columns:
      - YearsExperience: Number of years of experience (0 to 40).
      - EducationLevel: Education level represented as integers 1 to 4.
      - Salary: The salary computed as a function of experience and education level with added noise.

    Args:
        filename (str): The name of the CSV file to be created.
        num_rows (int): The number of rows to generate (default is 1000).
    """
    np.random.seed(42)  # For reproducibility

    # Generate random years of experience between 0 and 40
    years_experience = np.random.randint(0, 41, size=num_rows)

    # Generate random education levels with defined probabilities:
    # High School (1): 40%, Bachelor's (2): 30%, Master's (3): 20%, PhD (4): 10%
    education_levels = np.random.choice(
        [1, 2, 3, 4], size=num_rows, p=[0.4, 0.3, 0.2, 0.1]
    )

    # Define base salaries and experience bonuses for each education level
    base_salary = {1: 30000, 2: 40000, 3: 50000, 4: 60000}
    bonus_per_year = {1: 1000, 2: 1500, 3: 2000, 4: 2500}

    # Calculate salary for each row with some random noise
    salaries = []
    for exp, edu in zip(years_experience, education_levels):
        noise = np.random.normal(0, 5000)  # noise with standard deviation of 5000
        salary = base_salary[edu] + bonus_per_year[edu] * exp + noise
        salaries.append(salary)

    # Create the DataFrame and save it as a CSV file
    data = {
        "YearsExperience": years_experience,
        "EducationLevel": education_levels,
        "Salary": salaries,
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    generate_salary_csv("../data/salaries.csv")
