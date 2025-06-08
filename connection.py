from pymongo import MongoClient

# Replace with your actual URI
MONGO_URI = "mongodb+srv://yashmalviya2304:yash0539s@cluster0.enyverb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = MongoClient(MONGO_URI)

# Choose database and collection
db = client["lex"]
users_collection = db["users"]

# Example: insert a user
users_collection.insert_one({"username": "admin", "password": "secure123"})
