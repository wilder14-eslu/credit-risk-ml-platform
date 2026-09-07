CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    applicant_id VARCHAR(100),
    probability DOUBLE PRECISION NOT NULL,
    decision VARCHAR(30) NOT NULL,
    explanation JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS outcomes (
    id BIGSERIAL PRIMARY KEY,
    applicant_id VARCHAR(100) NOT NULL,
    actual_default BOOLEAN NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_applicant_id ON predictions (applicant_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_applicant_id ON outcomes (applicant_id);
