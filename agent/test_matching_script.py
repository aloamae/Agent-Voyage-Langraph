#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour débugger le matching de l'agent voyage
Usage: python test_matching.py
"""

import sys
sys.path.insert(0, '.')

from graph import VOYAGES, match_voyage
from typing import Dict, Optional

def print_voyage_details(voyage: Dict):
    """Affiche les détails d'un voyage de manière formatée."""
    print(f"    📋 Détails:")
    print(f"       - Plage: {voyage['plage']}")
    print(f"       - Montagne: {voyage['montagne']}")
    print(f"       - Ville: {voyage['ville']}")
    print(f"       - Sport: {voyage['sport']}")
    print(f"       - Détente: {voyage['detente']}")
    print(f"       - PMR: {voyage['acces_handicap']}")

def test_matching(test_name: str, criteres: Dict[str, Optional[bool]]):
    """
    Teste le matching avec des critères donnés.
    """
    print("\n" + "="*70)
    print(f"🧪 {test_name}")
    print("="*70)
    
    print("\n📊 Critères testés:")
    criteres_str = []
    for k, v in criteres.items():
        if v is True:
            criteres_str.append(f"  ✅ {k}: OUI")
        elif v is False:
            criteres_str.append(f"  ❌ {k}: NON")
        else:
            criteres_str.append(f"  ➖ {k}: (ignoré)")
    
    for line in criteres_str:
        print(line)
    
    print("\n🔍 Résultats du matching:")
    print("-" * 70)
    
    matches = []
    for i, voyage in enumerate(VOYAGES, 1):
        result = match_voyage(voyage, criteres)
        
        if result:
            print(f"\n{i}. ✅ MATCH: {voyage['nom']}")
            print_voyage_details(voyage)
            matches.append(voyage)
        else:
            print(f"\n{i}. ❌ Pas de match: {voyage['nom']}")
            
            # Analyser pourquoi ça ne match pas
            blocages = []
            for crit, val in criteres.items():
                if val is not None and crit in voyage:
                    if voyage[crit] != val:
                        blocages.append(
                            f"{crit} (demandé={val}, voyage={voyage[crit]})"
                        )
            
            if blocages:
                print(f"    🚫 Bloqué par: {', '.join(blocages)}")
    
    print("\n" + "="*70)
    print(f"📈 RÉSUMÉ: {len(matches)}/{len(VOYAGES)} voyage(s) correspondent")
    
    if matches:
        print(f"🎯 Premier match (proposé à l'utilisateur): {matches[0]['nom']}")
    else:
        print("⚠️  Aucun voyage ne correspond → Message 'assouplir critères'")
    
    print("="*70)
    
    return matches

def main():
    """Lance tous les tests de matching."""
    print("\n" + "🚀 " * 20)
    print("TESTS DE MATCHING - AGENT VOYAGE")
    print("🚀 " * 20 + "\n")
    
    # -------------------------------------------------------------------------
    # Test 1: Recherche simple - Plage uniquement
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 1: Recherche simple - Plage",
        {
            "plage": True,
            "montagne": None,
            "ville": None,
            "sport": None,
            "detente": None,
            "acces_handicap": None,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 2: Exclusivité stricte - Montagne seulement
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 2: Exclusivité stricte - Montagne (plage/ville=False)",
        {
            "plage": False,
            "montagne": True,
            "ville": False,
            "sport": None,
            "detente": None,
            "acces_handicap": None,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 3: Double critère - Sport + Montagne
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 3: Double critère - Sport + Montagne",
        {
            "plage": False,
            "montagne": True,
            "ville": False,
            "sport": True,
            "detente": False,
            "acces_handicap": None,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 4: Détente + PMR
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 4: Détente + Accessibilité PMR",
        {
            "plage": None,
            "montagne": None,
            "ville": None,
            "sport": False,
            "detente": True,
            "acces_handicap": True,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 5: Heuristique Montpellier (plage + ville, montagne=False)
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 5: Heuristique Montpellier (plage + ville)",
        {
            "plage": True,
            "montagne": False,
            "ville": True,
            "sport": None,
            "detente": None,
            "acces_handicap": None,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 6: Critères restrictifs (aucun match attendu)
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 6: Critères restrictifs - Plage + Sport + Montagne",
        {
            "plage": True,
            "montagne": True,  # Contradictoire avec plage (exclusivité)
            "ville": None,
            "sport": True,
            "detente": None,
            "acces_handicap": None,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 7: Ville uniquement (exclusivité)
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 7: Ville uniquement (exclusivité)",
        {
            "plage": False,
            "montagne": False,
            "ville": True,
            "sport": None,
            "detente": None,
            "acces_handicap": None,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 8: Tous critères à None (devrait retourner tous les voyages)
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 8: Aucun critère (tous None) - Attendu: tous matchent",
        {
            "plage": None,
            "montagne": None,
            "ville": None,
            "sport": None,
            "detente": None,
            "acces_handicap": None,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 9: Campagne (aucun thème géographique)
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 9: Campagne (plage=False, montagne=False, ville=False)",
        {
            "plage": False,
            "montagne": False,
            "ville": False,
            "sport": None,
            "detente": True,
            "acces_handicap": None,
        }
    )
    
    # -------------------------------------------------------------------------
    # Test 10: Sport sans détente
    # -------------------------------------------------------------------------
    test_matching(
        "TEST 10: Sport uniquement (détente=False)",
        {
            "plage": None,
            "montagne": None,
            "ville": None,
            "sport": True,
            "detente": False,
            "acces_handicap": None,
        }
    )
    
    print("\n" + "✅ " * 20)
    print("TESTS TERMINÉS")
    print("✅ " * 20 + "\n")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\n❌ ERREUR: Impossible d'importer depuis graph.py")
        print(f"   Détail: {e}")
        print(f"\n💡 Solution: Assure-toi que graph.py est dans le même dossier\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
