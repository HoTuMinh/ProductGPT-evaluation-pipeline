"""
Database models for evaluation logging and results storage
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()


class EvaluationRun(Base):
    """Store information about each evaluation run"""
    __tablename__ = "evaluation_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user = Column(String(100), nullable=False)
    input_file_name = Column(String(255), nullable=False)
    input_file_size = Column(Integer)  # in bytes
    metrics_evaluated = Column(JSON)  # List of metrics
    total_rows = Column(Integer)
    status = Column(String(50), default="running")  # running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Results summary
    average_scores = Column(JSON)  # {metric: score}
    total_api_calls = Column(Integer, default=0)
    execution_time_seconds = Column(Float)
    
    # Configuration used
    llm_model = Column(String(100))
    batch_size = Column(Integer)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user": self.user,
            "input_file_name": self.input_file_name,
            "input_file_size": self.input_file_size,
            "metrics_evaluated": self.metrics_evaluated,
            "total_rows": self.total_rows,
            "status": self.status,
            "error_message": self.error_message,
            "average_scores": self.average_scores,
            "total_api_calls": self.total_api_calls,
            "execution_time_seconds": self.execution_time_seconds,
            "llm_model": self.llm_model,
            "batch_size": self.batch_size
        }


class EvaluationResult(Base):
    """Store detailed results for each row evaluation"""
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, nullable=False)  # Foreign key to evaluation_runs
    row_index = Column(Integer, nullable=False)
    
    # Input data
    question = Column(Text)
    response = Column(Text)
    benchmark_answer = Column(Text)
    
    # Evaluation results
    metric_name = Column(String(50))
    score = Column(Float)
    label = Column(String(50))  # positive, negative, etc.
    reasoning = Column(Text)
    
    # Human review
    human_reviewed = Column(Integer, default=0)  # 0 or 1 (boolean)
    human_score = Column(Float, nullable=True)
    human_label = Column(String(50), nullable=True)
    human_comment = Column(Text, nullable=True)
    review_timestamp = Column(DateTime, nullable=True)
    
    # Metadata
    api_call_duration_ms = Column(Float)
    confidence = Column(Float)  # If available from LLM
    
    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "row_index": self.row_index,
            "question": self.question,
            "response": self.response,
            "benchmark_answer": self.benchmark_answer,
            "metric_name": self.metric_name,
            "score": self.score,
            "label": self.label,
            "reasoning": self.reasoning,
            "human_reviewed": bool(self.human_reviewed),
            "human_score": self.human_score,
            "human_label": self.human_label,
            "human_comment": self.human_comment,
            "review_timestamp": self.review_timestamp.isoformat() if self.review_timestamp else None,
            "api_call_duration_ms": self.api_call_duration_ms,
            "confidence": self.confidence
        }


class APIUsageLog(Base):
    """Track API usage for monitoring and cost management"""
    __tablename__ = "api_usage_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    run_id = Column(Integer)
    
    provider = Column(String(50))  # gemini, openai, etc.
    model = Column(String(100))
    
    # Token usage
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)
    
    # Cost estimation (if available)
    estimated_cost_usd = Column(Float, nullable=True)
    
    # Request details
    latency_ms = Column(Float)
    success = Column(Integer, default=1)  # 0 or 1
    error_message = Column(Text, nullable=True)


class Database:
    """Database manager class"""
    
    def __init__(self, db_path: str = "data/logs.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def create_evaluation_run(self, user: str, input_file_name: str, **kwargs):
        """Create a new evaluation run record"""
        session = self.get_session()
        try:
            run = EvaluationRun(
                user=user,
                input_file_name=input_file_name,
                **kwargs
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run.id
        finally:
            session.close()
    
    def update_evaluation_run(self, run_id: int, **kwargs):
        """Update an evaluation run record"""
        session = self.get_session()
        try:
            run = session.query(EvaluationRun).filter_by(id=run_id).first()
            if run:
                for key, value in kwargs.items():
                    setattr(run, key, value)
                session.commit()
        finally:
            session.close()
    
    def add_evaluation_result(self, run_id: int, row_index: int, **kwargs):
        """Add an evaluation result"""
        session = self.get_session()
        try:
            result = EvaluationResult(
                run_id=run_id,
                row_index=row_index,
                **kwargs
            )
            session.add(result)
            session.commit()
            return result.id
        finally:
            session.close()
    
    def log_api_usage(self, run_id: int, provider: str, model: str, **kwargs):
        """Log API usage"""
        session = self.get_session()
        try:
            log = APIUsageLog(
                run_id=run_id,
                provider=provider,
                model=model,
                **kwargs
            )
            session.add(log)
            session.commit()
        finally:
            session.close()
    
    def get_evaluation_runs(self, user: str = None, limit: int = 100):
        """Get recent evaluation runs"""
        session = self.get_session()
        try:
            query = session.query(EvaluationRun)
            if user:
                query = query.filter_by(user=user)
            runs = query.order_by(EvaluationRun.timestamp.desc()).limit(limit).all()
            return [run.to_dict() for run in runs]
        finally:
            session.close()
    
    def get_evaluation_results(self, run_id: int):
        """Get all results for a specific run"""
        session = self.get_session()
        try:
            results = session.query(EvaluationResult).filter_by(run_id=run_id).all()
            return [result.to_dict() for result in results]
        finally:
            session.close()
    
    def update_human_review(self, result_id: int, human_score: float, human_label: str, human_comment: str = None):
        """Update human review for a specific result"""
        session = self.get_session()
        try:
            result = session.query(EvaluationResult).filter_by(id=result_id).first()
            if result:
                result.human_reviewed = 1
                result.human_score = human_score
                result.human_label = human_label
                result.human_comment = human_comment
                result.review_timestamp = datetime.utcnow()
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def get_review_statistics(self, run_id: int, metric_name: str, threshold: float = 0.7):
        """Get review statistics for a run and metric"""
        session = self.get_session()
        try:
            results = session.query(EvaluationResult).filter_by(
                run_id=run_id,
                metric_name=metric_name
            ).all()
            
            total = len(results)
            reviewed = sum(1 for r in results if r.human_reviewed)
            
            # Original stats (LLM)
            llm_passes = sum(1 for r in results if r.score >= threshold)
            llm_pass_rate = (llm_passes / total * 100) if total > 0 else 0
            
            # Human review stats
            human_passes = 0
            agreements = 0
            
            for r in results:
                if r.human_reviewed:
                    # Use human score if reviewed
                    if r.human_score >= threshold:
                        human_passes += 1
                    # Check agreement
                    llm_pass = r.score >= threshold
                    human_pass = r.human_score >= threshold
                    if llm_pass == human_pass:
                        agreements += 1
                else:
                    # Use LLM score if not reviewed
                    if r.score >= threshold:
                        human_passes += 1
            
            human_pass_rate = (human_passes / total * 100) if total > 0 else 0
            agreement_rate = (agreements / reviewed * 100) if reviewed > 0 else 0
            
            return {
                'total_samples': total,
                'reviewed_count': reviewed,
                'llm_pass_rate': llm_pass_rate,
                'human_pass_rate': human_pass_rate,
                'agreement_rate': agreement_rate,
                'pass_rate_change': human_pass_rate - llm_pass_rate
            }
        finally:
            session.close()
