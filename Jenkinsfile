pipeline {
        agent {
        docker {
            image 'docker:20.10.21-dind'
            args '-v /var/run/docker.sock:/var/run/docker.sock'
        }
    }

    environment {
        COMPOSE_FILE = 'docker-compose.yml'
    }
    stages {
        stage('Test Docker Access') {
            steps {
                sh 'docker version'
            }
        }

        stage('Install Docker Compose') {
            steps {
                sh '''
                    if ! command -v docker-compose >/dev/null 2>&1; then
                      echo "Installing Docker Compose..."
                      curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
                      chmod +x /usr/local/bin/docker-compose
                      ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose || true
                    else
                      echo "Docker Compose already installed"
                    fi
                '''
            }
        }
        stage('📦 Checkout Code') {
            steps {
                checkout scm
            }
        }
        stage('🐳 Build Docker Images') {
            steps {
                sh  'docker-compose build'
            }
        }
        stage('🚀 Start Service Backend') {
            steps {
                sh 'docker-compose up -d backend'
                sh 'sleep 60'
            }
        }
        stage('🚀 Start Service  Node.js') {
            steps {
                sh  'docker-compose up -d nodejs'
                sh  'sleep 5'  // Windows equivalent of sleep
            }
        }
        stage('🧪 Run Python Backend Tests') {
            steps {
                sh  'docker exec backend pytest backend/tests --disable-warnings --maxfail=1'
            }
        }

        stage('🧪 Run Node.js Tests') {
            steps {
                sh  'docker exec nodejs npm install'
                sh  'docker exec nodejs npm test'
            }
        }
        stage('🛑 Stop Services') {
            steps {
                sh  'docker-compose down'
            }
        }
    }

    post {
        always {
            echo '🧹 Cleaning up containers and volumes...'
            sh  'docker-compose down -v'
        }
        success {
            echo '✅ All tests passed! Nice job!'
        }
        failure {
            echo '❌ Some tests failed! Check logs above.'
        }
    }
}
