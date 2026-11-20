# Installer Docker

Tout le TP tourne dans Docker : le projet se lance de la même façon sur toutes les machines.
Il vous faut **Docker** et **Docker Compose** (inclus dans Docker Desktop).

## macOS

1. Si vous n'avez pas Homebrew, installez-le (ouvrez le Terminal) :

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   À la fin, exécutez les deux lignes `eval ...` que l'installateur affiche (« Next steps »).

2. Installez Docker Desktop :

   ```bash
   brew install --cask docker
   ```

   Sans Homebrew : téléchargez Docker Desktop sur <https://www.docker.com/products/docker-desktop>
   (choisissez **Apple Silicon** pour un Mac M1/M2/M3/M4, **Intel** sinon), puis glissez-le dans Applications.

3. Lancez **Docker Desktop** (Applications), acceptez les conditions et attendez que la baleine
   de la barre de menus soit stable (« Docker Desktop is running »).

## Windows (10 ou 11)

1. Ouvrez **PowerShell en administrateur** (clic droit sur le menu Démarrer → « Terminal (administrateur) ») et installez WSL 2 :

   ```powershell
   wsl --install
   ```

   **Redémarrez** l'ordinateur si on vous le demande. Au redémarrage, une fenêtre Ubuntu s'ouvre : choisissez un nom d'utilisateur et un mot de passe.

2. Installez Docker Desktop :

   ```powershell
   winget install -e --id Docker.DockerDesktop
   ```

   Sans winget : téléchargez Docker Desktop sur <https://www.docker.com/products/docker-desktop>.
   Pendant l'installation, gardez cochée l'option **« Use WSL 2 instead of Hyper-V »**.

3. Lancez **Docker Desktop** et attendez qu'il indique « Engine running ».
   Dans *Settings → Resources → WSL integration*, activez l'intégration avec **Ubuntu**.

4. Travaillez ensuite dans un terminal **Ubuntu (WSL)** ou **Git Bash** pour les commandes du TP.

## Linux (Ubuntu, Debian, Fedora...)

1. Installez Docker Engine avec le script officiel :

   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   ```

2. Autorisez votre utilisateur à utiliser Docker sans `sudo`, puis **fermez et rouvrez votre session** :

   ```bash
   sudo usermod -aG docker $USER
   ```

3. Vérifiez que le service tourne : `sudo systemctl enable --now docker`.

Documentation détaillée par distribution : <https://docs.docker.com/engine/install/>.

## Vérifier l'installation

Dans un terminal :

```bash
docker --version              # Docker version 2x.x...
docker compose version        # Docker Compose version v2.x...
docker run --rm hello-world   # affiche « Hello from Docker! »
```

Si les trois commandes répondent, Docker est prêt. Téléchargez ensuite l'image de la base de données du TP :

```bash
docker pull pgvector/pgvector:pg16
```

## En cas de problème

| Symptôme | Solution |
|---|---|
| `Cannot connect to the Docker daemon` / `docker` ne répond pas | Docker Desktop n'est pas démarré : lancez-le et attendez qu'il soit stable. Sous Linux : `sudo systemctl start docker`. |
| `permission denied ... docker.sock` (Linux) | Vous n'avez pas refermé votre session après `usermod -aG docker` : déconnectez-vous et reconnectez-vous. |
| `docker compose` : commande inconnue | Docker trop ancien : mettez à jour Docker Desktop (ou installez le paquet `docker-compose-plugin` sous Linux). |
| Windows : « WSL 2 installation is incomplete » | Exécutez `wsl --update` dans PowerShell administrateur, puis redémarrez Docker Desktop. |
| Windows : « Virtualization must be enabled » | Activez la virtualisation (VT-x / AMD-V / SVM) dans le BIOS de l'ordinateur. |
| `port is already allocated` au lancement du TP | Un autre programme utilise le port (5180, 8010 ou 5433) : fermez-le ou arrêtez l'ancien conteneur (`docker ps` puis `docker stop <nom>`). |
