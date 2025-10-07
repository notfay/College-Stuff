package model;
import state.ChefIdle;
import state.State;

public class Chef {
	
	private String chefName;
	private int chefAge;
	private State state;
	
	public Chef(String chefName, int chefAge) {
		super();
		this.chefName = chefName;
		this.chefAge = chefAge;
		this.state = new ChefIdle(this);
	}

	public String getChefName() {
		return chefName;
	}

	public void setChefName(String chefName) {
		this.chefName = chefName;
	}

	public int getChefAge() {
		return chefAge;
	}

	public void setChefAge(int chefAge) {
		this.chefAge = chefAge;
	}

	public State getState() {
		return state;
	}

	public void setState(State state) {
		this.state = state;
	}
	
	
}
