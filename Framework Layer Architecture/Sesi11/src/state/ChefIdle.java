package state;

import model.Chef;

public class ChefIdle extends State{

	public ChefIdle(Chef chef) {
		super(chef);
		// TODO Auto-generated constructor stub
		System.out.println("Chef: " + chef.getChefName() + " is available");
	}

	@Override
	public void voidchangeState() {
		chef.setState(new ChefCooking(chef));
	}
	
	
}
