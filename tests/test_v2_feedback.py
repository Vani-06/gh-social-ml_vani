import uuid
from types import SimpleNamespace

from feedback.v2 import OrderedFeedbackApplier


class FakeQdrant:
    def __init__(self, user_id, repo_id, last=0):
        self.user_id = user_id
        self.repo_id = repo_id
        self.last = last
        self.upserts = []

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        if ids[0] == self.user_id:
            return [SimpleNamespace(id=self.user_id, vector=[1.0, 0.0], payload={"last_feedback_version": self.last})]
        return [SimpleNamespace(id=self.repo_id, vector={"repo_embedding": [0.0, 1.0]}, payload={})]

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


def event(user_id, repo_id, version):
    return {"event_id": str(uuid.uuid4()), "user_id": user_id, "repo_id": repo_id,
            "feedback_version": str(version), "event_type": "like", "dwell_ms": ""}


def test_feedback_applies_version_with_vector_in_one_upsert():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    result = OrderedFeedbackApplier(client).apply(event(user_id, repo_id, 1))
    assert result.status == "applied"
    point = client.upserts[0]["points"][0]
    assert point.payload["last_feedback_version"] == 1


def test_feedback_skips_duplicate_and_holds_version_gap():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    duplicate = OrderedFeedbackApplier(FakeQdrant(user_id, repo_id, last=2)).apply(event(user_id, repo_id, 2))
    gap_client = FakeQdrant(user_id, repo_id, last=2)
    gap = OrderedFeedbackApplier(gap_client).apply(event(user_id, repo_id, 4))
    assert duplicate.status == "duplicate"
    assert gap.status == "gap"
    assert gap_client.upserts == []
