# Ultimate AI Intent Classification Project

This project is devoted to a Flask HTTP service aming to provide the classification of intents from text, utilizing the ATIS dataset as the core dataset for training Transformer model.
The corresponding AI/ML implementation, research, and results can be found under `ml` directory and in detail described in `ml/README.md`.

While the dataset consists of English texts, the service is able to detect intents correctly for some languages (see example below) 😊.

Below, you will find details about the Flask service and instructions on how to run it.

## Local API Deployment
### 1. Install Docker:
> brew install docker
> 
> brew install docker-compose
### 2. Download Transformer model
Download `model.onnx` file from Google Drive ([model link](https://drive.google.com/file/d/10FirXm0jxhX2_aT6I3gM2Gs1-OdgswCS/view?usp=sharing) 📦🔗) and move it under `ml/models/all-mpnet-base-v2-tuned` directory

### 3. Build and run service:
In the project root directory run the following command in your terminal:
> make

That is it! Now you can send request to the Flask API on http://127.0.0.1:8080 🎉

# API Documentation
This API classifies user intents from text using a neural network model and offers two endpoints:

### 1. `/intent` - Intent Classification

- **POST**: Classify intents from text.
- **Request**: JSON body with `"text"` field.
- **Response**: JSON array of predicted top-k intents. See the example response below.
#### Request JSON
- **text** `string`: Text to be classified.

_Example Request:_
```json
{
 "text": "find me a flight that flies from memphis to tacoma"
}
```

#### Prediction JSON

- **label** `string`: The name of the predicted intent.
- **confidence** `float`: The probability score for the predicted intent.

_Example Response:_
```json
{
 "intents": [
   {
     "label": "flight",
     "confidence": 0.73
   },
   {
     "label": "aircraft",
     "confidence": 0.12
   },
   {
     "label": "capacity",
     "confidence": 0.03
   }
 ]
}
```

### 2. `/ready` - Service Readiness

- **GET**: Check if the model is ready.

_Example Response:_
```
OK
```
Or
```
Not ready
```

# Examples or Request-Response
#### English:
```json
Request JSON:
{
 "text": "Does Chicago Airport offer transportation from the airport to the downtown area?"
}

Response JSON:
{
    "intents": [
        {
            "confidence": 0.9949638843536377,
            "label": "ground_service"
        },
        {
            "confidence": 0.0007102342206053436,
            "label": "ground_service+ground_fare"
        },
        {
            "confidence": 0.0005169900832697749,
            "label": "abbreviation"
        }
    ]
}
```
A bit of my curiosity to challenge the model. Here, I provide examples of intent classification using text in various languages, note that the model was not fine-tuned with non-English texts. The same request from above has been translated into different languages using Google Translator. And of course we cannot assume that the model will correctly classify all non-English requests. 
#### German:
```json
{
 "text": "Bietet der Flughafen Chicago einen Transport vom Flughafen in die Innenstadt an?"
}

{
    "intents": [
        {
            "confidence": 0.9946973323822021,
            "label": "ground_service"
        },
        {
            "confidence": 0.0006867669872008264,
            "label": "ground_service+ground_fare"
        },
        {
            "confidence": 0.0006053699180483818,
            "label": "abbreviation"
        }
    ]
}
```
#### French:
```json
{
 "text": "L'aéroport de Chicago propose-t-il un transport de l'aéroport au centre-ville?"
}
{
    "intents": [
        {
            "confidence": 0.9948626160621643,
            "label": "ground_service"
        },
        {
            "confidence": 0.0006891299854032695,
            "label": "flight"
        },
        {
            "confidence": 0.0006123234634287655,
            "label": "ground_service+ground_fare"
        }
    ]
}
```