# InspectLine

[![CI](https://github.com/Julionores/inspectline/actions/workflows/ci.yml/badge.svg)](https://github.com/Julionores/inspectline/actions/workflows/ci.yml)

Un détecteur d'objets pour le contrôle qualité industriel : localiser **et** identifier
plusieurs pièces géométriques (cercle, carré, triangle, étoile) sur un tapis de convoyeur,
par fine-tuning ciblé d'un **Faster R-CNN** pré-entraîné sur COCO.

> Projet réalisé par **Junior Tsafack Megnekeu** ([blog.jtmcloud.com](https://blog.jtmcloud.com) ·
> [GitHub](https://github.com/Julionores) ·
> [LinkedIn](https://www.linkedin.com/in/junior-tsafack-megnekeu-b673151b9)) — pièce d'un
> portfolio technique orienté Machine Learning. Voir aussi
> [`gradientforge`](https://github.com/Julionores/gradientforge),
> [`radar-risque-impaye`](https://github.com/Julionores/radar-risque-impaye),
> [`collecte-agricole-planner`](https://github.com/Julionores/collecte-agricole-planner),
> [`ticket-tide`](https://github.com/Julionores/ticket-tide),
> [`runbook-rag`](https://github.com/Julionores/runbook-rag),
> [`agent-matching-recrutement`](https://github.com/Julionores/agent-matching-recrutement), et
> l'ensemble du portfolio :
> [`devsecops-pipeline-reference`](https://github.com/Julionores/devsecops-pipeline-reference),
> [`securebank-api`](https://github.com/Julionores/securebank-api),
> [`postgresql-ha-repmgr`](https://github.com/Julionores/postgresql-ha-repmgr),
> [`iso27001-isms-toolkit`](https://github.com/Julionores/iso27001-isms-toolkit),
> [`homelab-attaque-detection`](https://github.com/Julionores/homelab-attaque-detection),
> [`dynamodb-streams-cdc-pipeline`](https://github.com/Julionores/dynamodb-streams-cdc-pipeline),
> [`aws-troubleshooting-challenge`](https://github.com/Julionores/aws-troubleshooting-challenge),
> [`s3-cross-region-replication`](https://github.com/Julionores/s3-cross-region-replication),
> [`aws-alb-deployment-patterns`](https://github.com/Julionores/aws-alb-deployment-patterns),
> [`aws-vpc-connectivity-patterns`](https://github.com/Julionores/aws-vpc-connectivity-patterns) et
> [`mcp-odoo-toolkit`](https://github.com/Julionores/mcp-odoo-toolkit).
> Ce projet accompagne le module 7 de mon
> [cours Machine Learning & Deep Learning](https://blog.jtmcloud.com/machine-learning/07-vision-par-ordinateur/).

## Le scénario

Une ligne de production doit trier automatiquement des pièces géométriques déposées sur un
tapis de convoyeur, à partir d'une caméra fixe. Contrairement à la classification d'image
entière, le nombre de pièces par image varie : il faut à la fois **localiser** chaque pièce
et l'**identifier**.

## Un jeu de données 100% synthétique, sans dépendance externe

```python
from inspectline import generate_dataset

train = generate_dataset(200, size=160, seed=1)
```

Les images sont générées par code (formes, couleurs, positions et tailles aléatoires sur un
fond texturé) plutôt que téléchargées depuis un service tiers : **zéro dépendance externe
fragile**, un dataset **entièrement reproductible**, et un contrôle total de la difficulté.

## Avant / après fine-tuning

```bash
python examples/finetune_demo.py
```

```
=== Avant fine-tuning ===
Detections (score>0.5): 0

Parametres entrainables: 14,530,660 / 18,945,604
epoch 1/3 - loss moyenne: 0.7154
epoch 2/3 - loss moyenne: 0.4524
epoch 3/3 - loss moyenne: 0.4804
temps d'entrainement: 53.0s

=== Apres fine-tuning ===
Detections (score>0.5): 3
Formes predites: ['etoile', 'triangle', 'etoile']
Vraies formes: ['triangle', 'etoile', 'etoile']

=== mAP sur le jeu de validation ===
mAP (IoU 0.5:0.95): 0.8023
mAP@0.5: 1.0
```

Le détecteur COCO tel quel ne reconnaît **aucune** pièce comme un objet (0 détection) : nos
formes géométriques plates n'ont rien à voir avec les 91 catégories de COCO. Après un
fine-tuning ciblé — geler le backbone, ne réentraîner que la tête de classification — sur
seulement 200 images et 3 epochs (53 secondes sur CPU), le modèle détecte et classe
correctement les pièces, avec un mAP@0.5 parfait.

## Ce que le package garantit

- `build_model` + `freeze_backbone` : la tête de classification est adaptée au nombre de
  classes voulu, et le backbone reste gelé (`test_freeze_backbone_disables_gradients_on_backbone_only`).
- Le détecteur pré-entraîné, non fine-tuné, ne détecte **rien** sur ce domaine
  (`test_pretrained_model_detects_nothing_on_synthetic_shapes`) — le point pédagogique central
  du projet, vérifié comme test de non-régression.
- Un entraînement court sur un petit jeu de données produit bien un modèle qui détecte quelque
  chose (`test_fine_tuned_model_detects_shapes_after_short_training`) — un test d'intégration
  plus lent que les autres, qui entraîne réellement le modèle plutôt que de simuler le résultat.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

14 tests couvrent la génération de données, le pipeline Dataset/DataLoader, et le
comportement du modèle avant/après fine-tuning.

```
============================= 14 passed in 11.04s ==============================
```

## Installation

```bash
conda create -n inspectline python=3.11
conda activate inspectline
pip install -r requirements-dev.txt
```

## Limites assumées

- Données synthétiques, pas des photos réelles — un choix délibéré pour un pipeline rapide,
  reproductible et sans dépendance externe. L'architecture (Faster R-CNN allégé, tête
  remplacée, backbone gelé) s'applique à l'identique sur un jeu de données réel au format YOLO
  ou COCO.
- Aucune data augmentation, entraînement court (3 epochs), aucune classe « inconnue ».

## Licence

MIT — voir [`LICENSE`](LICENSE). Projet à but pédagogique et de démonstration.
