from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional 

class Student(BaseModel):
    name: str 
    age: Optional[int] = None  # age can be an integer or default value given to -> None 
    grade:str
    subjects: List[str] 
    email: EmailStr
    cgpa: float = Field(..., ge=0.0, le=4.0)  # cgpa must be between 0.0 and 4.0

# Pydantic tries to be forgiving (by default) and will attempt to coerce the string into an integer.  --->>>   "20" → 20 ✅ works fine.

# some time if i pass "32" in place of 32 it will also work fine. -> >> "32" → 32  pydantic is smart enough to handle this conversion.

# field functionality


new_student = {
    "name": "Alice Johnson",
    "age": 20,
    "grade": "A",
    "subjects": ["Math", "Science", "History"],
    "email": "sumit123@gmail.com",
    "cgpa" : 3 # cgpa must be between 0.0 and 4.0
}

student = Student(**new_student)
print(type(student))
print(student)
student_dict = student.model_dump()  # convert to dictionary
print(student_dict)
print(student_dict['name'])  # Accessing individual fields
student_json = student.model_dump_json()
print(preetty_json := student.model_dump_json(indent=4))  # pretty print json with indentation
