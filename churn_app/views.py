from django.shortcuts import render
import pickle
import os
import pandas as pd

from django.conf import settings



# Load trained pipeline model

model = pickle.load(
    open(
        os.path.join(
            settings.BASE_DIR,
            "ml/churn_model.pkl"
        ),
        "rb"
    )
)




def predict_churn(request):


    if request.method == "POST":


        # Collect customer details

        customer = {


            "gender":
            request.POST.get("gender"),


            "SeniorCitizen":
            int(request.POST.get("senior_citizen",0)),


            "Partner":
            request.POST.get("partner"),


            "Dependents":
            request.POST.get("dependents"),


            "tenure":
            int(request.POST.get("tenure")),


            "PhoneService":
            request.POST.get("phone_service"),


            "MultipleLines":
            request.POST.get("multiple_lines"),


            "InternetService":
            request.POST.get("internet_service"),


            "OnlineSecurity":
            request.POST.get("online_security"),


            "OnlineBackup":
            request.POST.get("online_backup"),


            "DeviceProtection":
            request.POST.get("device_protection"),


            "TechSupport":
            request.POST.get("tech_support"),


            "StreamingTV":
            request.POST.get("streaming_tv"),


            "StreamingMovies":
            request.POST.get("streaming_movies"),


            "Contract":
            request.POST.get("contract"),


            "PaperlessBilling":
            request.POST.get("paperless_billing"),


            "PaymentMethod":
            request.POST.get("payment_method"),


            "MonthlyCharges":
            float(request.POST.get("monthly_charge")),


            "TotalCharges":
            float(request.POST.get("total_charges"))

        }




        # Convert into dataframe

        input_data = pd.DataFrame(
            [customer]
        )




        # Prediction

        prediction = model.predict(
            input_data
        )


        probability = model.predict_proba(
            input_data
        )[0][1]



        probability = round(
            probability*100,
            2
        )




        # Prediction result

        if prediction[0] == 1:

            result = "Customer will Churn"


        else:

            result = "Customer will Stay"





        # -----------------------------
        # Customer Behaviour Explanation
        # -----------------------------


        reasons = []



        if customer["Contract"] == "Month-to-month":

            reasons.append(
                "Month-to-month contract increases churn possibility"
            )



        if customer["tenure"] < 12:

            reasons.append(
                "Low customer tenure indicates a new customer"
            )



        if customer["PaymentMethod"] == "Electronic check":

            reasons.append(
                "Electronic check customers show higher churn tendency"
            )



        if customer["TechSupport"] == "No":

            reasons.append(
                "No technical support service may reduce satisfaction"
            )



        if customer["OnlineSecurity"] == "No":

            reasons.append(
                "No online security service increases churn risk"
            )



        if customer["MonthlyCharges"] > 80:

            reasons.append(
                "High monthly charges may affect customer retention"
            )



        if len(reasons) == 0:

            reasons.append(
                "Customer profile shows stable behaviour"
            )






        # -----------------------------
        # Risk Level + Suggestions
        # -----------------------------


        if probability < 30:


            risk = "LOW"


            suggestions = [

                "Maintain current service quality",

                "Provide loyalty rewards",

                "Offer premium benefits"

            ]



        elif probability < 70:


            risk = "MEDIUM"


            suggestions = [

                "Offer loyalty discount",

                "Encourage yearly contract",

                "Recommend additional services",

                "Improve customer engagement"

            ]



        else:


            risk = "HIGH"


            suggestions = [

                "Contact customer immediately",

                "Provide retention discount",

                "Offer better contract plans",

                "Check customer service issues",

                "Provide technical support package"

            ]





        context = {


            "result":
            result,


            "probability":
            probability,


            "risk":
            risk,


            "reasons":
            reasons,


            "suggestions":
            suggestions


        }




        return render(
            request,
            "predict.html",
            context
        )



    return render(
        request,
        "predict.html"
    )