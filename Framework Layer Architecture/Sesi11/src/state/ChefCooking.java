package state;

import model.Chef;

public class ChefCooking extends State{

	public ChefCooking(Chef chef) {
		super(chef);
		// TODO Auto-generated constructor stub
		System.out.println("Chef: " + chef.getChefName() + " is cooking");
	}
	
	@Override
	public void voidchangeState() {
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		chef.setState(new ChefDelivery(chef));
		
	}
	
	
}
