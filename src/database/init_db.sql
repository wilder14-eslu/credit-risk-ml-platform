CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    applicant_id VARCHAR(100),
    probability DOUBLE PRECISION NOT NULL,
    decision VARCHAR(30) NOT NULL,
    explanation JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
