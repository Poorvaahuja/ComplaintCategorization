from pydantic import BaseModel
class Complaint(BaseModel):
    complaint: str