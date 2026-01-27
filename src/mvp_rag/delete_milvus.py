from pymilvus import connections, utility

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
utility.has_collection("Physical_Design")
utility.drop_collection("Physical_Design")
print("✅ Collection deleted successfully")
