"""Policy Engine for the ShopConnect Object-Centric Intelligence Environment.

Implements deterministic pre-commit and post-commit validation rules:
- Dietary restrictions compliance (UserPreferences.diet_tags)
- Allergen safety validation (UserPreferences.allergies)
- Weekly budget enforcement (UserPreferences.weekly_budget)
- State invariant checks (non-target meal preservation, plan version increment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from shopconnect_env.domain.models import Ingredient, MealPlan, Recipe


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of a policy evaluation."""

    allowed: bool
    rule_name: str
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


def normalize_allergen(name: str) -> str:
    """Normalize allergen name and common aliases to canonical form."""
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "peanut": "peanuts",
        "peanuts": "peanuts",
        "milk": "dairy",
        "dairy": "dairy",
        "cow_milk": "dairy",
        "gluten": "gluten",
        "wheat": "wheat",
        "egg": "eggs",
        "eggs": "eggs",
        "soy": "soy",
        "soya": "soy",
        "soybean": "soy",
        "fish": "fish",
        "seafood": "fish",
        "tree_nut": "tree_nuts",
        "tree_nuts": "tree_nuts",
        "nut": "tree_nuts",
        "nuts": "tree_nuts",
    }
    return alias_map.get(cleaned, cleaned)


def allergens_conflict(user_allergy: str, ingredient_allergen: str) -> bool:
    """Check if a user allergy matches an ingredient allergen classification.

    Medical and Biological Distinction Rules:
    - Peanuts vs Tree Nuts: Peanuts are legumes (Arachis hypogaea), distinct from
      botanical tree nuts (almonds, walnuts, cashews, etc.). They are never treated as identical.
    - Milk vs Dairy: 'milk' and 'dairy' are treated as synonymous terms for mammalian milk proteins.
    - Wheat vs Gluten: Directional hierarchy without unjustified equivalence claims.
      Wheat inherently contains gluten; therefore, a user with a gluten constraint (u == 'gluten')
      must avoid wheat (i == 'wheat').
      Conversely, a user with a specific wheat allergy (u == 'wheat') is allergic to wheat-specific
      proteins; non-wheat gluten-containing grains (e.g. pure oats, barley, rye) are not claimed
      to be wheat.
    """
    u = normalize_allergen(user_allergy)
    i = normalize_allergen(ingredient_allergen)
    if u == i:
        return True

    # Directional hierarchy: wheat inherently contains gluten
    if u == "gluten" and i == "wheat":
        return True

    # Synonymous family: dairy and milk are identical
    if u in {"dairy", "milk"} and i in {"dairy", "milk"}:
        return True

    # Synonymous family: tree nuts aliases
    if u in {"tree_nuts", "nuts"} and i in {"tree_nuts", "nuts"}:
        return True

    return False


def normalize_diet_tag(tag: str) -> str:
    """Normalize dietary filter tags (e.g. 'Vegetarian' -> 'vegetarian')."""
    return tag.strip().lower().replace("_", "-").replace(" ", "-")


class PolicyEngine:
    """Deterministic business rule and safety verifier.

    LIMITATION NOTICE:
        1. Synthetic allergen metadata is an offline research simulation and NOT a substitute
           for actual food packaging verification or manufacturer allergen declarations.
        2. Cross-contamination (e.g. shared factory lines) is not modeled.
        3. Budget checks reflect projected costs of all *currently scheduled* meals. If a plan
           has unfilled slots, the calculated cost represents partial expenditure, not a full week.
    """

    def __init__(
        self,
        catalog: Dict[str, Recipe],
        ingredients: Dict[str, Ingredient],
    ) -> None:
        self.catalog = catalog
        self.ingredients = ingredients

    def validate_dietary_restrictions(self, plan: MealPlan) -> PolicyDecision:
        """Verify that every scheduled meal in the plan satisfies configured dietary tags."""
        configured_tags = [
            normalize_diet_tag(t) for t in plan.preferences.diet_tags if t.strip()
        ]
        if not configured_tags:
            return PolicyDecision(
                allowed=True,
                rule_name="dietary_restrictions",
                reason="No dietary restrictions configured.",
            )

        for meal in plan.meals.values():
            if meal.recipe_id not in self.catalog:
                return PolicyDecision(
                    allowed=False,
                    rule_name="dietary_restrictions",
                    reason=f"Recipe '{meal.recipe_id}' scheduled for '{meal.slot_key}' not found in catalog.",
                )
            recipe = self.catalog[meal.recipe_id]

            # Normalize recipe tags and include implicit tags (e.g. vegan implies vegetarian & dairy-free)
            recipe_tags = {normalize_diet_tag(t) for t in recipe.tags}
            if "vegan" in recipe_tags:
                recipe_tags.add("vegetarian")
                recipe_tags.add("dairy-free")

            for required_tag in configured_tags:
                if required_tag not in recipe_tags:
                    return PolicyDecision(
                        allowed=False,
                        rule_name="dietary_restrictions",
                        reason=(
                            f"Recipe '{recipe.name}' ({recipe.recipe_id}) scheduled for '{meal.slot_key}' "
                            f"violates dietary restriction: missing required tag '{required_tag}'."
                        ),
                        details={
                            "slot_key": meal.slot_key,
                            "recipe_id": recipe.recipe_id,
                            "missing_tag": required_tag,
                            "recipe_tags": list(recipe.tags),
                        },
                    )

        return PolicyDecision(
            allowed=True,
            rule_name="dietary_restrictions",
            reason="All scheduled meals satisfy configured dietary restrictions.",
        )

    def validate_allergens(self, plan: MealPlan) -> PolicyDecision:
        """Verify that no scheduled recipe contains ingredients conflicting with user allergies.

        CONSERVATIVE SAFETY RULE:
            If an ingredient has unknown allergen metadata (allergens is None) while the user
            has active allergy constraints, the recipe is conservatively rejected.
        """
        user_allergies = [
            a.strip() for a in plan.preferences.allergies if a.strip()
        ]
        if not user_allergies:
            return PolicyDecision(
                allowed=True,
                rule_name="allergen_safety",
                reason="No user allergies configured.",
            )

        for meal in plan.meals.values():
            if meal.recipe_id not in self.catalog:
                return PolicyDecision(
                    allowed=False,
                    rule_name="allergen_safety",
                    reason=f"Recipe '{meal.recipe_id}' not found in catalog.",
                )
            recipe = self.catalog[meal.recipe_id]

            for req in recipe.requirements:
                if req.ingredient_id not in self.ingredients:
                    return PolicyDecision(
                        allowed=False,
                        rule_name="allergen_safety",
                        reason=f"Ingredient '{req.ingredient_id}' not found in ingredient catalog.",
                    )
                ingredient = self.ingredients[req.ingredient_id]

                # Case 1: Unknown allergen metadata -> Conservative rejection
                if ingredient.allergens is None:
                    return PolicyDecision(
                        allowed=False,
                        rule_name="allergen_safety",
                        reason=(
                            f"Conservative allergen rejection: Ingredient '{ingredient.name}' ({ingredient.ingredient_id}) "
                            f"in recipe '{recipe.name}' has unknown allergen metadata (None) while active allergies "
                            f"are configured: {user_allergies}."
                        ),
                        details={
                            "slot_key": meal.slot_key,
                            "recipe_id": recipe.recipe_id,
                            "ingredient_id": ingredient.ingredient_id,
                            "allergens": None,
                        },
                    )

                # Case 2: Explicit allergen metadata -> Conflict checking
                for user_allergy in user_allergies:
                    for ing_allergen in ingredient.allergens:
                        if allergens_conflict(user_allergy, ing_allergen):
                            return PolicyDecision(
                                allowed=False,
                                rule_name="allergen_safety",
                                reason=(
                                    f"Allergen conflict detected: Recipe '{recipe.name}' scheduled for '{meal.slot_key}' "
                                    f"contains '{ingredient.name}' ({ingredient.ingredient_id}) with allergen '{ing_allergen}', "
                                    f"violating configured user allergy '{user_allergy}'."
                                ),
                                details={
                                    "slot_key": meal.slot_key,
                                    "recipe_id": recipe.recipe_id,
                                    "ingredient_id": ingredient.ingredient_id,
                                    "matched_allergen": ing_allergen,
                                    "user_allergy": user_allergy,
                                },
                            )

        return PolicyDecision(
            allowed=True,
            rule_name="allergen_safety",
            reason="All scheduled meals pass allergen safety verification.",
        )

    def validate_budget(self, plan: MealPlan) -> PolicyDecision:
        """Verify that the projected total cost of scheduled meals does not exceed the weekly budget."""
        try:
            projected_cost = plan.calculate_total_cost(self.catalog)
        except KeyError as e:
            return PolicyDecision(
                allowed=False,
                rule_name="weekly_budget",
                reason=f"Budget calculation failed due to missing recipe: {e}",
            )

        budget = plan.preferences.weekly_budget

        details = {
            "projected_cost": str(projected_cost),
            "weekly_budget": str(budget),
            "is_complete_plan": plan.is_complete(),
            "scheduled_slots_count": len(plan.meals),
        }

        if projected_cost > budget:
            return PolicyDecision(
                allowed=False,
                rule_name="weekly_budget",
                reason=(
                    f"Projected weekly plan cost (${projected_cost:.2f}) exceeds configured "
                    f"weekly budget limit (${budget:.2f})."
                ),
                details=details,
            )

        return PolicyDecision(
            allowed=True,
            rule_name="weekly_budget",
            reason=f"Projected plan cost (${projected_cost:.2f}) is within budget (${budget:.2f}).",
            details=details,
        )

    def validate_invariants(
        self,
        original_plan: MealPlan,
        candidate_plan: MealPlan,
        target_slot_key: Optional[str] = None,
    ) -> PolicyDecision:
        """Verify strict postconditions and state integrity between original and candidate plans."""
        # 1. Preferences must be identical
        if candidate_plan.preferences != original_plan.preferences:
            return PolicyDecision(
                allowed=False,
                rule_name="state_invariants",
                reason="User preferences were unexpectedly mutated during operation.",
            )

        # 2. Plan version must increment exactly once
        if candidate_plan.plan_version != original_plan.plan_version + 1:
            return PolicyDecision(
                allowed=False,
                rule_name="state_invariants",
                reason=(
                    f"Invalid plan version transition: expected {original_plan.plan_version + 1}, "
                    f"got {candidate_plan.plan_version}."
                ),
            )

        # 3. Non-target meal slots must remain byte-identical
        for slot_key, original_meal in original_plan.meals.items():
            if slot_key != target_slot_key:
                if slot_key not in candidate_plan.meals:
                    return PolicyDecision(
                        allowed=False,
                        rule_name="state_invariants",
                        reason=f"Non-target meal slot '{slot_key}' was deleted in candidate plan.",
                    )
                candidate_meal = candidate_plan.meals[slot_key]
                if candidate_meal != original_meal:
                    return PolicyDecision(
                        allowed=False,
                        rule_name="state_invariants",
                        reason=(
                            f"Non-target meal slot '{slot_key}' was unexpectedly modified. "
                            f"Original: ({original_meal.recipe_id}, {original_meal.servings} servings) vs "
                            f"Candidate: ({candidate_meal.recipe_id}, {candidate_meal.servings} servings)."
                        ),
                    )

        # 4. No unauthorized extra slots added
        for slot_key in candidate_plan.meals.keys():
            if slot_key != target_slot_key and slot_key not in original_plan.meals:
                return PolicyDecision(
                    allowed=False,
                    rule_name="state_invariants",
                    reason=f"Unauthorized new slot '{slot_key}' added to candidate plan.",
                )

        return PolicyDecision(
            allowed=True,
            rule_name="state_invariants",
            reason="All state invariants and non-target meal slots verified intact.",
        )

    def validate_candidate(
        self,
        original_plan: MealPlan,
        candidate_plan: MealPlan,
        target_slot_key: Optional[str] = None,
    ) -> PolicyDecision:
        """Execute all policy rules and invariant checks against a candidate plan in priority order."""
        # Check invariants first
        inv_decision = self.validate_invariants(original_plan, candidate_plan, target_slot_key)
        if not inv_decision.allowed:
            return inv_decision

        # Check dietary restrictions
        diet_decision = self.validate_dietary_restrictions(candidate_plan)
        if not diet_decision.allowed:
            return diet_decision

        # Check allergen safety
        allergen_decision = self.validate_allergens(candidate_plan)
        if not allergen_decision.allowed:
            return allergen_decision

        # Check budget limits
        budget_decision = self.validate_budget(candidate_plan)
        if not budget_decision.allowed:
            return budget_decision

        return PolicyDecision(
            allowed=True,
            rule_name="all_passed",
            reason="Candidate plan passed all invariant, dietary, allergen, and budget policy checks.",
        )
