s := false

git:
	git pull origin 2021
	git add .
	git commit -m "$m"
	git push origin HEAD -o ci.variable="CONDOR_SUBMIT=$s"
