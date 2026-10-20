
from flask import Flask, render_template, request
from model_utils import predict_movie_roi

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        movie_details = {
            "budget": float(request.form["budget"]),
            "runtime": float(request.form["runtime"]),
            "franchise": int(request.form["franchise"]),
            "release_year": int(request.form["release_year"]),
            "release_month": int(request.form["release_month"]),
            "original_language": request.form["original_language"].strip(),
            "release_season": request.form["release_season"].strip(),
            "director": request.form["director"].strip(),
            "genres": request.form.getlist("genres"),
            "keywords": request.form.get("keywords", ""),
            "companies": request.form.get("companies", ""),
            "countries": request.form.get("countries", ""),
            "languages": request.form.get("languages", ""),
            "overview": request.form.get("overview", "").strip(),
        }

        predicted_roi = predict_movie_roi(**movie_details)

        return render_template(
            "index.html",
            prediction=round(predicted_roi, 4),
            prediction_percentage=round(predicted_roi * 100, 2)
        )

    except (ValueError, KeyError) as error:
        return render_template(
            "index.html",
            error=f"Please check your inputs. {error}"
        ), 400

    except Exception:
        app.logger.exception("Movie ROI prediction failed")
        return render_template(
            "index.html",
            error="Prediction failed. Please check the movie details and try again."
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
