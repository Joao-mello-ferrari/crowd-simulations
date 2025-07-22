.PHONY: sim1 sim2 sim3 sim4 sim5 sim6 sim7

sim1:
	@echo "Simulating Biocrowds with Boids - 40 agents - 2 groups - with obstacles"
	python3 boids_biocrowds.py \
		--radius 0.3 \
		--num-agents 40 \
		--num-groups 2 \
		--perception 1 \
		--use-obstacles \
		--marker-density 18 \
		--agent-types biocrowds,boids

sim2:
	@echo "Simulating Biocrowds with Boids - 40 agents - 4 groups - with obstacles"
	python3 boids_biocrowds.py \
		--radius 0.3 \
		--num-agents 40 \
		--num-groups 4 \
		--perception 1 \
		--use-obstacles \
		--marker-density 18 \
		--agent-types biocrowds,boids

sim3:
	@echo "Simulating Biocrowds with Boids - 40 agents - 2 groups - no obstacles"
	python3 boids_biocrowds.py \
		--radius 0.3 \
		--num-agents 40 \
		--num-groups 2 \
		--perception 1 \
		--marker-density 18 \
		--agent-types biocrowds,boids

sim4:
	@echo "Simulating Biocrowds with Boids - 40 agents - 4 groups - no obstacles"
	python3 boids_biocrowds.py \
		--radius 0.3 \
		--num-agents 40 \
		--num-groups 4 \
		--perception 1 \
		--marker-density 18 \
		--agent-types biocrowds,boids

sim5:
	@echo "Simulating Boids - 40 agents - 4 groups - no obstacles"
	python3 boids_biocrowds.py \
		--radius 0.3 \
		--num-agents 40 \
		--num-groups 4 \
		--perception 1 \
		--marker-density 18 \
		--agent-types boids

sim6:
	@echo "Simulating Biocrowds - 40 agents - 4 groups - no obstacles"
	python3 boids_biocrowds.py \
		--radius 0.2 \
		--num-agents 40 \
		--num-groups 4 \
		--perception 1 \
		--marker-density 18 \
		--agent-types biocrowds

sim7:
	@echo "Simulating Biocrowds - 40 agents - 4 groups - no obstacles - allowing biocrowds collisions"
	python3 boids_biocrowds.py \
		--radius 0.2 \
		--num-agents 40 \
		--num-groups 4 \
		--perception 1 \
		--marker-density 18 \
		--agent-types biocrowds \
		--allow-biocrowds-collision

sim8:
	@echo "Simulating Biocrowds - 60 agents - 2 groups - no obstacles - allowing biocrowds collisions"
	python3 boids_biocrowds.py \
		--radius 0.3 \
		--num-agents 60 \
		--num-groups 2 \
		--perception 1 \
		--marker-density 18 \
		--agent-types biocrowds \
		--allow-biocrowds-collision
