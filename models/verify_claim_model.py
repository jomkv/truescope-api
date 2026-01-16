from pydantic import BaseModel

class VerifyClaimModel(BaseModel):
    claim: str
    limit: int = 10
    use_fallback: bool = True