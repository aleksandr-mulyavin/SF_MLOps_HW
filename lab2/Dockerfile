FROM jenkins/jenkins:jdk17

# Проброс портов (уже есть в исходном образе)
EXPOSE 8080 50000

# Установка питона
USER root
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv

# Запуск скрипта от пользователя jenkins
USER jenkins
CMD ["jenkins.sh"]