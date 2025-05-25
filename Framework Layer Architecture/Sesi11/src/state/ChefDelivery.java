package state;

import model.Chef;

public class ChefDelivery extends State{

	public ChefDelivery(Chef chef) {
		super(chef);
		System.out.println("Chef: " + chef.getChefName() + " is delivery");
	}

	@Override
	public void voidchangeState() {
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		chef.setState(new ChefIdle(chef));
		
	}
	

}
