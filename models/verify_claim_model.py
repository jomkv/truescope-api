from pydantic import BaseModel


class VerifyConfigModel(BaseModel):
    maxEvidence: int | None = None
    useNonFactcheck: bool | None = None


class VerifyClaimModel(BaseModel):
    claim: str
    config: VerifyConfigModel | None = None
