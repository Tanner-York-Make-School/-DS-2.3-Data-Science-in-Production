# STEP 1: Download the base image. Optimized for python.
FROM python:3.7-slim-buster

# STEP 2: Add requiremnts text file
ADD requirements.txt /requirements.txt

# STEP 3: Install required pyhton dependencies from requirements file
RUN pip install -r requirements.txt

# STEP 4: Add the source code of the current directory into docker container
ADD ./ /app

# STEP 5: Set working directory
WORKDIR /app

# STEP 6: Expose ports Flask is running on 
EXPOSE 8000

# STEP 7: Run Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"] 