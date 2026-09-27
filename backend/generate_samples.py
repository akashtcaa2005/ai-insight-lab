"""
One-off script that generates the sample CSV datasets bundled with the
app (data/sample/). Run again any time you want to regenerate them:

    python generate_samples.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = "app/sample_data"


def house_prices(n=400):
    area = rng.normal(1800, 500, n).clip(400, 5000)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    age = rng.integers(0, 60, n)
    location = rng.choice(["Downtown", "Suburb", "Rural", "Waterfront"], n, p=[0.3, 0.4, 0.2, 0.1])
    location_premium = pd.Series(location).map(
        {"Downtown": 1.35, "Waterfront": 1.6, "Suburb": 1.0, "Rural": 0.75}
    ).values
    base_price = area * 120 + bedrooms * 8000 + bathrooms * 6000 - age * 400
    price = (base_price * location_premium + rng.normal(0, 15000, n)).clip(30000)
    df = pd.DataFrame(
        {
            "area_sqft": area.round(0),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "location": location,
            "price": price.round(0),
        }
    )
    mask = rng.random(n) < 0.03
    df.loc[mask, "bathrooms"] = np.nan
    df.to_csv(f"{OUT}/house_prices.csv", index=False)


def student_performance(n=350):
    study_hours = rng.normal(4, 2, n).clip(0, 12)
    attendance = rng.normal(80, 12, n).clip(30, 100)
    sleep_hours = rng.normal(6.5, 1.3, n).clip(3, 10)
    prior_score = rng.normal(65, 15, n).clip(20, 100)
    score = (
        study_hours * 4.2
        + attendance * 0.35
        + sleep_hours * 1.5
        + prior_score * 0.25
        + rng.normal(0, 6, n)
    ).clip(0, 100)
    passed = (score >= 50).astype(int)
    df = pd.DataFrame(
        {
            "study_hours_per_day": study_hours.round(1),
            "attendance_pct": attendance.round(1),
            "sleep_hours": sleep_hours.round(1),
            "prior_exam_score": prior_score.round(1),
            "final_score": score.round(1),
            "passed": passed,
        }
    )
    df.to_csv(f"{OUT}/student_performance.csv", index=False)


def customer_churn(n=500):
    tenure_months = rng.integers(1, 72, n)
    monthly_charges = rng.normal(70, 25, n).clip(15, 150)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    support_calls = rng.poisson(1.5, n)
    contract_factor = pd.Series(contract).map(
        {"Month-to-month": 0.45, "One year": 0.15, "Two year": 0.05}
    ).values
    churn_prob = (
        contract_factor
        + (support_calls > 3) * 0.25
        + (tenure_months < 6) * 0.2
        + (monthly_charges > 100) * 0.1
    ).clip(0, 0.9)
    churn = (rng.random(n) < churn_prob).astype(int)
    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "contract_type": contract,
            "support_calls": support_calls,
            "churn": churn,
        }
    )
    df.to_csv(f"{OUT}/customer_churn.csv", index=False)


def employee_salary(n=300):
    experience = rng.integers(0, 25, n)
    education = rng.choice(["Bachelors", "Masters", "PhD"], n, p=[0.55, 0.35, 0.1])
    dept = rng.choice(["Engineering", "Sales", "Marketing", "Operations"], n)
    edu_bonus = pd.Series(education).map({"Bachelors": 0, "Masters": 12000, "PhD": 25000}).values
    dept_bonus = pd.Series(dept).map(
        {"Engineering": 15000, "Sales": 5000, "Marketing": 3000, "Operations": 0}
    ).values
    salary = 40000 + experience * 2800 + edu_bonus + dept_bonus + rng.normal(0, 6000, n)
    df = pd.DataFrame(
        {
            "years_experience": experience,
            "education_level": education,
            "department": dept,
            "salary": salary.round(0).clip(28000),
        }
    )
    df.to_csv(f"{OUT}/employee_salary.csv", index=False)


def iris():
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target_names[df["target"]]
    df.columns = [c.replace(" (cm)", "_cm").replace(" ", "_") for c in df.columns]
    df.to_csv(f"{OUT}/iris_species.csv", index=False)


if __name__ == "__main__":
    house_prices()
    student_performance()
    customer_churn()
    employee_salary()
    iris()
    print("Sample datasets generated in", OUT)
