import math
import random
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FeedAssemblySystem:
    @staticmethod
    def process_feed_assembly(candidates: List[Dict[str, Any]], target_size: int = 15) -> List[str]:
        """
        Executes Freshness Injection and Exploration Injection sequentially
        on the pre-ranked top-15 JSON payload from the ranking system.
        """
        if target_size <= 0:
            return []

        current_time = datetime.now(timezone.utc)
        ranked_candidates = [dict(item) for item in candidates]
        
        # --- PART 1: FRESHNESS INJECTION ---
        for item in ranked_candidates:
            # Check explicitly for None so valid 0.0 scores are preserved
            base_score = item.get('final_score')
            if base_score is None:
                base_score = item.get('score')
            if base_score is None:
                base_score = 0.5
            item['score'] = base_score
                
            raw_created_at = item.get('created_at')
            if not raw_created_at:
                continue
                
            try:
                if isinstance(raw_created_at, str):
                    clean_timestamp = raw_created_at.replace('Z', '+00:00')
                    created_date = datetime.fromisoformat(clean_timestamp)
                elif isinstance(raw_created_at, datetime):
                    created_date = raw_created_at
                else:
                    continue

                if created_date.tzinfo is None:
                    created_date = created_date.replace(tzinfo=timezone.utc)

                # Add a guard before line 40:
                if created_date > current_time:
                    age_hours = 48.0  # Treat it as old/neutral rather than brand new
                else:
                    age_hours = max(0.0, (current_time - created_date).total_seconds() / 3600.0)
                    if age_hours < 48.0:
                        boost = 0.25 * (1.1 / (1.0 + math.log1p(age_hours)))
                        item['score'] += boost
            except Exception as e:
                logger.error(f"Freshness parsing failed for repo {item.get('repo_id')}: {e}")
                continue

        # Re-sort the 15 repos after applying freshness boosts
        ranked_candidates.sort(key=lambda x: x.get('score') if x.get('score') is not None else 0.5, reverse=True)

        target_count = min(target_size, len(ranked_candidates))
        final_pool = ranked_candidates[:target_count]

        # --- PART 2: EXPLORATION INJECTION ---
        # Keep the top tier stable and shuffle a dynamic discovery tail.
        if target_count >= 3:
            explore_count = max(1, target_count // 3)
            split_index = target_count - explore_count
            exploit_tier = final_pool[:split_index]
            explore_tier = final_pool[split_index:]
            
            random.shuffle(explore_tier)
            final_pool = exploit_tier + explore_tier

        # Strip internal temporary scores and return clean ordered string IDs
        ordered_ids = []
        for item in final_pool:
            repo_id = item.get('repo_id')
            if not repo_id:
                raise ValueError("Feed candidate missing required repo_id")
            ordered_ids.append(str(repo_id))

        return ordered_ids
