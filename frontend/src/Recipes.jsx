// Fiches des recettes citées dans une réponse du chatbot : vignettes (photo, durée, personnes) et fiche complète au clic.
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { getRecipe, recipeImageUrl } from "./api.js";

const EMOJI = { "Entrées": "🥗", "Plats": "🍲", "Desserts": "🍰", "Boissons": "🍹" };

function Photo({ recipe, className }) {
  const [failed, setFailed] = useState(false);
  if (failed) return <div className={`${className} placeholder`}>{EMOJI[recipe.category] || "🍽️"}</div>;
  return (
    <img className={className} src={recipeImageUrl(recipe.document_id, recipe.chunk_index)} alt=""
         loading="lazy" onError={() => setFailed(true)} />
  );
}

export function RecipeCard({ recipe: r, servings }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className="recipe-card" onClick={() => setOpen(true)} title={r.title} aria-label={`Voir la recette : ${r.title}`}>
        <Photo recipe={r} className="recipe-thumb" />
      </button>
      {open && createPortal(<RecipeModal recipe={r} servings={servings} onClose={() => setOpen(false)} />, document.body)}
    </>
  );
}

export function RecipeCards({ recipes, servings }) {
  if (!recipes?.length) return null;
  return (
    <div className="recipe-cards">
      {recipes.map((r) => <RecipeCard key={`${r.document_id}-${r.chunk_index}`} recipe={r} servings={servings} />)}
    </div>
  );
}

function RecipeModal({ recipe, servings: wanted, onClose }) {
  // Ouverte directement pour le nombre de personnes de la conversation (sinon celui de la recette)
  const [servings, setServings] = useState(wanted || recipe.servings || null);
  const [full, setFull] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getRecipe(recipe.document_id, recipe.chunk_index, servings).then(setFull).catch((e) => setError(e.message));
  }, [recipe.document_id, recipe.chunk_index, servings]);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const r = full || recipe;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal recipe-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={recipe.title}>
        <div className="modal-header">
          <h2>{recipe.title}</h2>
          <button type="button" className="link" onClick={onClose} aria-label="Fermer">✕</button>
        </div>
        <div className="modal-body">
          <Photo recipe={recipe} className="recipe-hero" />
          {error && <p className="error-text">⚠️ {error}</p>}
          <div className="recipe-facts">
            {r.category && <span>{EMOJI[r.category] || "🍽️"} {r.category}</span>}
            {r.duration_text && <span>⏱ {r.duration_text}</span>}
            {servings && (
              <span className="servings">
                👥
                <button type="button" onClick={() => setServings(Math.max(1, servings - 1))} aria-label="Moins de personnes">−</button>
                {servings} pers.
                <button type="button" onClick={() => setServings(servings + 1)} aria-label="Plus de personnes">+</button>
              </span>
            )}
          </div>
          {full?.description && <p className="recipe-description">{full.description}</p>}
          {full ? (
            <>
              <h3>Ingrédients</h3>
              <ul>{full.ingredients.map((i, k) => <li key={k}>{i.raw}</li>)}</ul>
              {full.utensils?.length > 0 && (
                <>
                  <h3>Ustensiles</h3>
                  <ul className="utensils">{full.utensils.map((u, k) => <li key={k}>{u}</li>)}</ul>
                </>
              )}
              <h3>Préparation</h3>
              <ol>{full.steps.map((s, k) => <li key={k}>{s}</li>)}</ol>
              {full.source_url && (
                <p className="recipe-source">
                  <a href={full.source_url} target="_blank" rel="noreferrer">Voir la recette sur Marmiton ↗</a>
                </p>
              )}
            </>
          ) : (
            !error && <p className="hint">Chargement…</p>
          )}
        </div>
      </div>
    </div>
  );
}
