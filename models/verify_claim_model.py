from pydantic import BaseModel

class VerifyClaimModel(BaseModel):
    claim: str
    limit: int = 10
    exclude_id: str | None = None
    use_fallback: bool = True