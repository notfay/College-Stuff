package state;

import model.Chef;

public abstract class State {
	
	protected Chef chef;

	public State(Chef chef) {
		super();
		this.chef = chef;
	}
	
	public abstract void voidchangeState();
}
